import argparse
import csv
import importlib
import inspect
import os
import statistics
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure generator parameters, FLOPs, inference latency, and FPS. "
            "The discriminator and training-only loss networks are excluded."
        )
    )
    parser.add_argument("--model_name", default="Ours", help="name shown in the output table")
    parser.add_argument(
        "--network_module",
        default="models.networks_improved_PTM1",
        help="Python module that provides define_G",
    )
    parser.add_argument("--checkpoint", default="", help="optional generator checkpoint")
    parser.add_argument("--non_strict", action="store_true", help="allow missing checkpoint keys")
    parser.add_argument("--device", default="cuda:0", help="for example cuda:0 or cpu")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--guide_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--netG", default="bFT_resnet")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--norm", default="instance")
    parser.add_argument("--init_type", default="normal")
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--use_z", action="store_true")
    parser.add_argument("--z_nc", type=int, default=1)
    parser.add_argument(
        "--ftm_mode",
        choices=("none", "film", "spatial"),
        default="spatial",
        help="passed only when define_G supports this argument",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timing_iters", type=int, default=100)
    parser.add_argument("--timing_rounds", type=int, default=5)
    parser.add_argument(
        "--flops_backend",
        choices=("hooks", "profiler", "thop"),
        default="hooks",
        help=(
            "hooks counts Conv/ConvTranspose/Linear plus executed batched matrix multiplies; "
            "thop requires pip install thop"
        ),
    )
    parser.add_argument("--csv", default="complexity_results.csv", help="CSV output; empty disables it")
    return parser.parse_args()


def validate_args(args):
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.warmup < 0 or args.timing_iters <= 0 or args.timing_rounds <= 0:
        raise ValueError("warmup must be non-negative and timing counts must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")


def build_generator(args):
    network_module = importlib.import_module(args.network_module)
    define_g = network_module.define_G
    effective_guide_nc = args.guide_nc + args.z_nc if args.use_z else args.guide_nc
    kwargs = {
        "input_nc": args.input_nc,
        "guide_nc": effective_guide_nc,
        "output_nc": args.output_nc,
        "ngf": args.ngf,
        "netG": args.netG,
        "n_layers": args.n_layers,
        "norm": args.norm,
        "init_type": args.init_type,
        "init_gain": args.init_gain,
        # Build on CPU first to avoid DataParallel changing checkpoint keys.
        "gpu_ids": [],
    }
    if "ftm_mode" in inspect.signature(define_g).parameters:
        kwargs["ftm_mode"] = args.ftm_mode
    elif args.ftm_mode != "spatial":
        raise ValueError(
            "%s.define_G does not accept ftm_mode, so --ftm_mode %s cannot be applied"
            % (args.network_module, args.ftm_mode)
        )

    model = define_g(**kwargs)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=not args.non_strict)
    return model, effective_guide_nc


def load_checkpoint(model, checkpoint_path, strict):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "netG", "generator"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint does not contain a state dictionary")

    for prefix in ("module.", "netG.", "generator."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print("Checkpoint missing keys: %d" % len(result.missing_keys))
        print("Checkpoint unexpected keys: %d" % len(result.unexpected_keys))


def make_inputs(args, effective_guide_nc, device):
    shape_source = (args.batch_size, args.input_nc, args.height, args.width)
    shape_guide = (args.batch_size, effective_guide_nc, args.height, args.width)
    source = torch.randn(shape_source, device=device)
    target_guide = torch.randn(shape_guide, device=device)
    source_guide = torch.randn(shape_guide, device=device)
    if args.use_z:
        source_guide[:, args.guide_nc:] = 0
    return source, target_guide, source_guide


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def profile_flops_with_torch(model, inputs, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    total_flops = sum(float(event.flops or 0) for event in profiler.key_averages())
    if total_flops <= 0:
        raise RuntimeError(
            "PyTorch profiler returned zero FLOPs. Try --flops_backend thop after installing thop."
        )
    return total_flops


def profile_flops_with_hooks(model, inputs, device):
    layer_flops = [0.0]
    handles = []

    def convolution_hook(module, _inputs, output):
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
        kernel_height, kernel_width = module.kernel_size
        kernel_multiplications = (
            kernel_height * kernel_width * module.in_channels / module.groups
        )
        layer_flops[0] += float(output_tensor.numel()) * kernel_multiplications * 2.0

    def transposed_convolution_hook(module, module_inputs, _output):
        input_tensor = module_inputs[0]
        kernel_height, kernel_width = module.kernel_size
        output_channels_per_group = module.out_channels / module.groups
        kernel_multiplications = (
            kernel_height * kernel_width * output_channels_per_group
        )
        # Each input element is expanded through the transposed-convolution
        # kernel. Using the output spatial size would overcount stride-2 layers.
        layer_flops[0] += float(input_tensor.numel()) * kernel_multiplications * 2.0

    def linear_hook(module, _inputs, output):
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
        layer_flops[0] += float(output_tensor.numel()) * module.in_features * 2.0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(transposed_convolution_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        with torch.inference_mode():
            with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                model(*inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    # Convolution and linear work is counted by hooks. Add only functional
    # batched matrix multiplications used by the attention module here.
    bmm_flops = 0.0
    bmm_calls = 0
    for event in profiler.key_averages():
        event_name = getattr(event, "key", "")
        if event_name == "aten::bmm":
            bmm_flops += float(event.flops or 0)
            bmm_calls += int(event.count)
    if bmm_calls and bmm_flops == 0:
        print("Warning: this PyTorch version detected aten::bmm but did not estimate its FLOPs.")
    return layer_flops[0] + bmm_flops, bmm_calls, bmm_flops


def profile_flops_with_thop(model, inputs):
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError("THOP is not installed. Run: pip install thop") from exc

    macs, _ = profile(model, inputs=inputs, verbose=False)
    # This script uses the convention that one multiply-accumulate is two FLOPs.
    return float(macs) * 2.0


def measure_latency(model, inputs, device, warmup, timing_iters, timing_rounds):
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        round_batch_ms = []
        for _ in range(timing_rounds):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(timing_iters):
                    model(*inputs)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms = start.elapsed_time(end)
            else:
                start_time = time.perf_counter()
                for _ in range(timing_iters):
                    model(*inputs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            round_batch_ms.append(elapsed_ms / timing_iters)

    mean_batch_ms = statistics.mean(round_batch_ms)
    std_batch_ms = statistics.pstdev(round_batch_ms)
    return mean_batch_ms, std_batch_ms


def print_markdown_table(row):
    headers = [
        "Model",
        "Parameters (M) down",
        "MACs (G) down",
        "FLOPs (G) down",
        "Time (ms/image) down",
        "FPS up",
    ]
    values = [
        row["Model"],
        "%.3f" % row["Parameters (M)"],
        "%.3f" % row["MACs (G)"],
        "%.3f" % row["FLOPs (G)"],
        "%.3f" % row["Inference Time (ms/image)"],
        "%.3f" % row["FPS"],
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    print("| " + " | ".join(values) + " |")


def append_csv(path, row):
    fieldnames = list(row.keys())
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as csv_file:
            existing_fields = next(csv.reader(csv_file), [])
        if existing_fields != fieldnames:
            raise ValueError(
                "CSV columns have changed. Use a new --csv path, for example "
                "complexity_results_macs.csv"
            )
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    model, effective_guide_nc = build_generator(args)
    model = model.to(device).eval()
    inputs = make_inputs(args, effective_guide_nc, device)

    # Validate the requested configuration before profiling or timing it.
    with torch.inference_mode():
        outputs = model(*inputs)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        print("Warning: expected two generator outputs, received %s" % type(outputs).__name__)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    parameters = count_parameters(model)
    if args.flops_backend == "hooks":
        flops, bmm_calls, bmm_flops = profile_flops_with_hooks(model, inputs, device)
        flops_note = (
            "module hooks for Conv/ConvTranspose/Linear plus %d executed bmm calls "
            "(%.3f GFLOPs)" % (bmm_calls, bmm_flops / 1e9)
        )
    elif args.flops_backend == "profiler":
        flops = profile_flops_with_torch(model, inputs, device)
        flops_note = "PyTorch profiler supported-operation count"
    else:
        flops = profile_flops_with_thop(model, inputs)
        flops_note = "THOP MAC count converted with 1 MAC = 2 FLOPs"

    batch_ms, batch_ms_std = measure_latency(
        model,
        inputs,
        device,
        args.warmup,
        args.timing_iters,
        args.timing_rounds,
    )
    flops_per_image = flops / args.batch_size
    macs_per_image = flops_per_image / 2.0
    image_ms = batch_ms / args.batch_size
    image_ms_std = batch_ms_std / args.batch_size
    fps = 1000.0 / image_ms

    row = {
        "Model": args.model_name,
        "Parameters (M)": parameters / 1e6,
        "MACs (G)": macs_per_image / 1e9,
        "FLOPs (G)": flops_per_image / 1e9,
        "Inference Time (ms/image)": image_ms,
        "FPS": fps,
        "Input": "%dx%d" % (args.height, args.width),
        "Batch Size": args.batch_size,
        "Device": str(device),
        "FLOPs Backend": args.flops_backend,
    }
    print_markdown_table(row)
    print("\nTiming standard deviation: %.3f ms/image" % image_ms_std)
    print("Operation count: %s." % flops_note)
    print("MAC/FLOPs convention: 1 MAC = 2 FLOPs.")
    print("Only generator G is included; discriminator D and loss networks are excluded.")
    print("PyTorch version: %s" % torch.__version__)
    if device.type == "cuda":
        print("GPU: %s" % torch.cuda.get_device_name(device))

    if args.csv:
        append_csv(args.csv, row)
        print("CSV saved to: %s" % os.path.abspath(args.csv))


if __name__ == "__main__":
    main()

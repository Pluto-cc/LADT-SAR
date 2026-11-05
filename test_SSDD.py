"""Generate SSDD test images with the trained GAN baselines.

The script uses the ``to`` image in a pair CSV to construct the conditional
mask.  It supports the generator checkpoints produced by train_gan_ssdd.py
and the custom cGAN implementation.
"""

import argparse
import csv
import importlib.util
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_TO_CHANNEL = {"land": 0, "sea": 1, "ship": 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image_dir",
        default="/home/王长启/Code/ASOC1/datasets/SSDD/SSDD_format_HBB/test",
    )
    parser.add_argument(
        "--annotation_dir",
        default="/home/王长启/Code/ASOC1/datasets/SSDD/SSDD_format_HBB/Annotations",
    )
    parser.add_argument(
        "--pair_csv",
        default="/home/王长启/Code/ASOC1/datasets/SSDD/market-pairs-test.csv",
    )
    parser.add_argument("--results_dir", default="./results_ssdd_test")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--z_nc", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["ours"])
    parser.add_argument("--ours_checkpoint", required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(module_name, file_path):
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError("model module does not exist: %s" % file_path)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def xml_float(element, name):
    value = element.findtext(name)
    if value is None:
        raise ValueError("missing XML field: %s" % name)
    return float(value.strip())


class PairTestMaskDataset(Dataset):
    """Create masks from target images listed in an external pair CSV."""

    def __init__(self, image_dir, annotation_dir, pair_csv, image_size):
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.annotation_dir = Path(annotation_dir).expanduser().resolve()
        self.pair_csv = Path(pair_csv).expanduser().resolve()
        self.image_size = int(image_size)
        for path, label in (
            (self.image_dir, "image directory"),
            (self.annotation_dir, "annotation directory"),
            (self.pair_csv, "pair CSV"),
        ):
            if not path.exists():
                raise FileNotFoundError("%s does not exist: %s" % (label, path))

        self.images = {
            path.name.lower(): path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        self.annotations = {
            path.stem.lower(): path
            for path in self.annotation_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".xml"
        }
        self.samples = []
        with self.pair_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "to" not in reader.fieldnames:
                raise ValueError("pair CSV must contain a 'to' column")
            for row_number, row in enumerate(reader, start=2):
                target_name = (row.get("to") or "").strip()
                image_path = self.images.get(target_name.lower())
                annotation_path = self.annotations.get(Path(target_name).stem.lower())
                if image_path is None:
                    raise FileNotFoundError(
                        "target image in row %d does not exist: %s"
                        % (row_number, target_name)
                    )
                if annotation_path is None:
                    raise FileNotFoundError(
                        "target annotation in row %d does not exist: %s"
                        % (row_number, target_name)
                    )
                self.samples.append(
                    {
                        "from": (row.get("from") or "").strip(),
                        "to": target_name,
                        "image": image_path,
                        "annotation": annotation_path,
                    }
                )
        if not self.samples:
            raise RuntimeError("no test pairs found in %s" % self.pair_csv)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _category(obj):
        name = (obj.findtext("name") or "").strip().lower()
        aliases = {
            "ground": "land",
            "shore": "land",
            "coast": "land",
            "land area": "land",
            "water": "sea",
            "ocean": "sea",
        }
        return aliases.get(name, name)

    def _scaled_point(self, x, y, source_width, source_height):
        target = self.image_size
        return (
            max(0.0, min(float(target - 1), float(x) / source_width * target)),
            max(0.0, min(float(target - 1), float(y) / source_height * target)),
        )

    def _polygon(self, obj, source_width, source_height):
        hbb = obj.find("bndbox")
        if hbb is not None:
            xmin = xml_float(hbb, "xmin")
            ymin = xml_float(hbb, "ymin")
            xmax = xml_float(hbb, "xmax")
            ymax = xml_float(hbb, "ymax")
            points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        else:
            obb = obj.find("robndbox")
            if obb is None:
                return None
            cx = xml_float(obb, "cx")
            cy = xml_float(obb, "cy")
            width = xml_float(obb, "w")
            height = xml_float(obb, "h")
            angle = xml_float(obb, "angle")
            cos_angle, sin_angle = math.cos(angle), math.sin(angle)
            points = []
            for dx, dy in (
                (-width / 2.0, -height / 2.0),
                (width / 2.0, -height / 2.0),
                (width / 2.0, height / 2.0),
                (-width / 2.0, height / 2.0),
            ):
                points.append(
                    (
                        cx + dx * cos_angle - dy * sin_angle,
                        cy + dx * sin_angle + dy * cos_angle,
                    )
                )
        return [
            self._scaled_point(x, y, source_width, source_height) for x, y in points
        ]

    def _mask(self, annotation_path, source_width, source_height):
        masks = [Image.new("L", (self.image_size, self.image_size), 0) for _ in range(3)]
        draws = [ImageDraw.Draw(mask) for mask in masks]
        root = ET.parse(str(annotation_path)).getroot()
        for obj in root.findall("object"):
            category = self._category(obj)
            channel = CLASS_TO_CHANNEL.get(category)
            polygon = self._polygon(obj, source_width, source_height)
            if channel is not None and polygon is not None:
                draws[channel].polygon(polygon, fill=255, outline=255)
        array = np.stack([np.asarray(mask, dtype=np.float32) / 255.0 for mask in masks])
        return torch.from_numpy(array.copy())

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(str(sample["image"])) as source:
            source_width, source_height = source.size
        mask = self._mask(sample["annotation"], source_width, source_height)
        return mask, sample["to"], index


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise TypeError("unsupported checkpoint object: %s" % type(checkpoint))
    for key in ("generator", "net_g", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("checkpoint does not contain a generator state_dict")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_generator(module, checkpoint_path, device, kind, z_nc=8):
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    if kind == "sargan":
        generator = module.define_G(3, 3, 16, "batch", False, "normal", 0.02, gpu_id=device)
    elif kind == "locsar":
        generator = module.define_G(3, 3, 16, "batch", False, "normal", 0.02, gpu_id=device)
    elif kind == "pix2pix":
        generator = module.define_G(3, 3, 64, "batch", False, "normal", 0.02, gpu_id=device)
    else:
        generator = module.ConditionalGenerator(
            mask_channels=3,
            output_channels=3,
            z_nc=z_nc,
            ngf=64,
            num_res_blocks=6,
        ).to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad = False
    return generator


def import_generators(sota_root):
    return {
        "sargan": load_module(
            "ssdd_test_sargan_networks",
            sota_root / "SARGAN" / "code" / "networks_stackhourglass_double_sf.py",
        ),
        "locsar": load_module(
            "ssdd_test_locsar_networks",
            sota_root / "LocSAR" / "code" / "NOSIM.py",
        ),
        "pix2pix": load_module(
            "ssdd_test_pix2pix_networks",
            sota_root / "Pix2Pix" / "code" / "networks.py",
        ),
        "cgan": load_module("ssdd_test_cgan", sota_root / "CGAN" / "cGAN.py"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    sota_root = Path(__file__).resolve().parent
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device("cuda:%d" % args.gpu_id)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dataset = PairTestMaskDataset(
        args.image_dir, args.annotation_dir, args.pair_csv, args.image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    modules = import_generators(sota_root)
    checkpoints = {
        "sargan": args.sargan_checkpoint,
        "locsar": args.locsar_checkpoint,
        "pix2pix": args.pix2pix_checkpoint,
        "cgan": args.cgan_checkpoint,
    }
    generators = {
        name: load_generator(
            modules[name], checkpoints[name], device, name, z_nc=args.z_nc
        )
        for name in args.models
    }

    results_root = Path(args.results_dir).expanduser().resolve()
    for name in args.models:
        (results_root / name).mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "from", "to"] + list(args.models))
        for masks, names, indices in tqdm(loader, desc="Generating SSDD test", ncols=120):
            masks = masks.to(device, non_blocking=True)
            outputs = {}
            with torch.no_grad():
                for name in args.models:
                    if name == "cgan":
                        noise_generator = torch.Generator(device=device).manual_seed(
                            args.seed + int(indices[0])
                        )
                        noise = torch.randn(
                            masks.size(0), args.z_nc, 1, 1,
                            generator=noise_generator, device=device
                        )
                        output = generators[name](masks, noise)
                    else:
                        output = generators[name](masks)
                    outputs[name] = output.add(1.0).div(2.0).clamp(0.0, 1.0)

            for batch_index, sample_index in enumerate(indices.tolist()):
                row = [sample_index, dataset.samples[sample_index]["from"], names[batch_index]]
                for name in args.models:
                    output_path = results_root / name / (
                        "%06d_%s" % (sample_index, Path(names[batch_index]).stem) + ".png"
                    )
                    save_image(outputs[name][batch_index], str(output_path))
                    row.append(str(output_path))
                writer.writerow(row)
    print("Generated %d pair-conditioned samples" % len(dataset))
    print("Results: %s" % results_root)


if __name__ == "__main__":
    main()

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11-OBB with a controlled random seed.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for this training run.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/SSDD.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="ultralytics/cfg/models/11/yolo11-obb.yaml", help="Model YAML path.")
    parser.add_argument("--name", default=None, help="Run name. Defaults to train_seed<seed>.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.name or f"train_seed{args.seed}"

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        workers=0,
        amp=False,
        imgsz=args.imgsz,
        device=args.device,
        pretrained=False,
        verbose=True,
        optimizer="AdamW",
        seed=args.seed,
        deterministic=True,
        project="runs/obb",
        name=run_name,
    )

    # Evaluate the selected best checkpoint on the validation split.
    best_weights = f"runs/obb/{run_name}/weights/best.pt"
    best_model = YOLO(best_weights)
    metrics = best_model.val(data=args.data, split="val", imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(f"Seed: {args.seed}")
    print(f"Validation metrics: {metrics.results_dict}")


if __name__ == "__main__":
    main()

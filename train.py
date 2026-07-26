import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 steel surface defect detector."
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练模型")
    parser.add_argument("--data", type=str, required=True, help="数据集YAML路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图片尺寸")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备，例如 cpu、mps、cuda 或 CUDA 编号 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from path_validation import (
        resolve_dataset_config_path,
        validate_dataset_config,
        validate_model_path,
    )

    for error in (validate_model_path(args.model), validate_dataset_config(args.data)):
        if error:
            print(f"❌ {error}", file=sys.stderr)
            return 2
    data_path = resolve_dataset_config_path(args.data)

    from ultralytics import YOLO
    from ultralytics.data import utils as data_utils
    import torch.multiprocessing

    data_utils.DATASETS_DIR = data_path.parent

    # Windows 下多进程保护
    torch.multiprocessing.freeze_support()

    try:
        print("🚀 初始化训练...")
        print(f"• 模型: {args.model}")
        print(f"• 数据: {args.data}")
        print(f"• 轮数: {args.epochs}")
        print(f"• 设备: {args.device or 'auto'}")

        model = YOLO(args.model)

        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": 2,  # Windows下如果报错，设为0
            "exist_ok": True,
            "name": "train_result",
        }
        if args.device:
            train_kwargs["device"] = args.device

        results = model.train(**train_kwargs)

        print("\n🎉 训练流程结束")
        print(f"💾 模型保存路径: {results.save_dir}")

    except Exception as e:
        print(f"\n❌ 训练中断: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

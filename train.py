import argparse
import sys

from path_validation import resolved_dataset_config_path


def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 steel surface defect detector.')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    parser.add_argument('--data', type=str, required=True, help='数据集YAML路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批大小')
    parser.add_argument('--imgsz', type=int, default=640, help='训练图片尺寸')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备，例如 cpu、mps、cuda 或 CUDA 编号 0'
    )
    parser.add_argument('--workers', type=int, default=2, help='数据加载进程数，Windows 报错时设为 0')
    parser.add_argument('--project', type=str, default='runs/detect', help='保存根目录')
    parser.add_argument('--name', type=str, default='train_result', help='实验名称，已存在时自动追加序号')
    parser.add_argument('--exist-ok', action='store_true', help='复用已存在的实验目录（会覆盖旧权重）')
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO
    import torch.multiprocessing

    # Windows 下多进程保护
    torch.multiprocessing.freeze_support()

    try:
        print("🚀 初始化训练...")
        print(f"• 模型: {args.model}")
        print(f"• 数据: {args.data}")
        print(f"• 轮数: {args.epochs}")
        print(f"• 设备: {args.device or 'auto'}")

        model = YOLO(args.model)

        with resolved_dataset_config_path(args.data) as dataset_config:
            train_kwargs = {
                'data': str(dataset_config),
                'epochs': args.epochs,
                'imgsz': args.imgsz,
                'batch': args.batch,
                'workers': args.workers,
                'project': args.project,
                'name': args.name,
                # 默认不复用旧目录：Ultralytics 会把新指标追加到旧 results.csv 并覆盖旧权重
                'exist_ok': args.exist_ok,
            }
            if args.device:
                train_kwargs['device'] = args.device

            results = model.train(**train_kwargs)

        print("\n🎉 训练流程结束")
        print(f"💾 模型保存路径: {results.save_dir}")

    except Exception as e:
        print(f"\n❌ 训练中断: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

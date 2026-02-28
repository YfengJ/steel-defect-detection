import argparse
import sys
import os
from ultralytics import YOLO
import torch.multiprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    parser.add_argument('--data', type=str, required=True, help='数据集YAML路径')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    return parser.parse_args()


def main():
    # Windows 下多进程保护
    torch.multiprocessing.freeze_support()

    args = parse_args()

    try:
        print(f"🚀 初始化训练...")
        print(f"• 模型: {args.model}")
        print(f"• 数据: {args.data}")
        print(f"• 轮数: {args.epochs}")

        model = YOLO(args.model)

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=2,  # Windows下如果报错，设为0
            exist_ok=True,
            name='train_result'
        )

        print("\n🎉 训练流程结束")
        print(f"💾 模型保存路径: {results.save_dir}")

    except Exception as e:
        print(f"\n❌ 训练中断: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
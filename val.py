import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a trained YOLOv8 steel defect detection model.')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, default=None, help='可选，覆盖模型中的data配置')
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()

    try:
        print(f"🔍 正在加载模型: {args.model}")
        model = YOLO(args.model)

        print("⏳ 开始验证数据集...")
        # 验证
        metrics = model.val(
            data=args.data,
            split='val',
            verbose=True
        )

        # 打印清晰的摘要供UI捕获
        print("\n" + "=" * 30)
        print("📊 验证结果摘要")
        print("=" * 30)
        print(f"mAP50-95 : {metrics.box.map:.4f}")
        print(f"mAP50    : {metrics.box.map50:.4f}")
        print(f"mAP75    : {metrics.box.map75:.4f}")
        print("-" * 30)

        # 如果有类别细分
        if hasattr(metrics.box, 'maps'):
            print("📈 各类别 mAP50-95:")
            names = model.names
            for i, score in enumerate(metrics.box.maps):
                cls_name = names[i] if names else str(i)
                print(f"  - {cls_name:<10}: {score:.4f}")
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"❌ 验证出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

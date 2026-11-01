import argparse
import json
import sys

from path_validation import resolved_dataset_config_path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a trained YOLOv8 steel defect detection model.')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, default=None, help='可选，覆盖模型中的data配置')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='验证划分')
    parser.add_argument('--device', type=str, default=None, help='验证设备，例如 cpu、mps、cuda 或 0')
    return parser.parse_args()


def metrics_summary(box, names):
    """Machine-readable summary consumed by the GUI (one JSON line)."""
    classes = []
    for i, cls_id in enumerate(box.ap_class_index):
        cls_id = int(cls_id)
        p, r, ap50, ap = box.class_result(i)
        classes.append({
            'id': cls_id,
            'name': names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
            'precision': float(p),
            'recall': float(r),
            'map50': float(ap50),
            'map': float(ap),
        })
    return {
        'precision': float(box.mp),
        'recall': float(box.mr),
        'map50': float(box.map50),
        'map75': float(box.map75),
        'map': float(box.map),
        'classes': classes,
    }


def main():
    args = parse_args()

    from ultralytics import YOLO
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()

    try:
        print(f"🔍 正在加载模型: {args.model}")
        model = YOLO(args.model)

        print("⏳ 开始验证数据集...")
        val_kwargs = {'split': args.split, 'verbose': True}
        if args.device:
            val_kwargs['device'] = args.device
        if args.data:
            with resolved_dataset_config_path(args.data) as dataset_config:
                metrics = model.val(data=str(dataset_config), **val_kwargs)
        else:
            metrics = model.val(**val_kwargs)

        # 打印清晰的摘要供UI捕获
        print("\n" + "=" * 30)
        print("📊 验证结果摘要")
        print("=" * 30)
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")
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
        print("METRICS_JSON " + json.dumps(metrics_summary(metrics.box, model.names), ensure_ascii=False))

    except Exception as e:
        print(f"❌ 验证出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

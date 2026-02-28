import argparse
import sys
from pathlib import Path
import time
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--source', type=str, required=True, help='图片/视频源')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--project', type=str, default='runs/detect', help='保存根目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    # 兼容性参数（虽然YOLOv8默认有，但显式声明防止报错）
    parser.add_argument('--save', action='store_true', help='保存图片')
    parser.add_argument('--save_txt', action='store_true', help='保存标签')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 准备路径
    project_dir = Path(args.project)
    save_dir = project_dir / args.name

    # 2. 加载模型
    try:
        print(f"🔮 加载模型: {args.model}")
        model = YOLO(args.model)

        # 3. 执行预测
        # UI 需要读取 labels 下的 txt 文件，且需要置信度，所以必须强制 save_txt=True, save_conf=True
        print(f"🖼️正在处理: {args.source}")
        start_t = time.time()

        results = model.predict(
            source=args.source,
            project=args.project,
            name=args.name,
            conf=args.conf,
            save=True,  # 强制保存图片
            save_txt=True,  # 强制保存TXT（UI生成报告需要）
            save_conf=True,  # 强制保存置信度（UI生成图表需要）
            exist_ok=True,  # 允许覆盖
            verbose=False  # 减少控制台刷屏
        )

        end_t = time.time()
        print(f"✅ 预测完成，耗时 {end_t - start_t:.2f}s")
        print(f"📂 结果已保存至: {save_dir.absolute()}")

    except Exception as e:
        print(f"❌ 预测发生错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
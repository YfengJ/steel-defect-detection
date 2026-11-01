import argparse
import math

# 中文字体候选（Windows / macOS / Linux），找不到时回退到 PIL 默认字体
FONT_CANDIDATES = (
    "simhei.ttf",
    "msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)


def load_font(size=20):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    print("⚠️ 未找到中文字体，使用系统默认字体")
    return ImageFont.load_default()


class VideoPredictor:
    @staticmethod
    def run(model_path, source, output_path="output.mp4", conf=0.25):
        """
        独立运行的视频预测函数
        """
        cap = None
        writer = None
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw

            model = YOLO(model_path)

            # 打开视频
            cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频源: {source}")

            # 获取属性
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if not math.isfinite(fps) or fps <= 0:
                fps = 25
            if w <= 0 or h <= 0:
                raise ValueError(f"视频尺寸无效: {w}x{h}")

            # 输出设置
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            if not writer.isOpened():
                raise OSError(f"无法创建输出视频: {output_path}。请检查目录、写入权限和视频编码支持。")

            font = load_font(20)

            print(f"▶ 开始处理视频，保存至: {output_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理
                results = model(frame, conf=conf, verbose=False)

                # 先用 OpenCV 画框，再一次性转换到 PIL 绘制中文标签，
                # 避免每个目标都整帧转换颜色空间
                labels = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        score = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        labels.append((x1, y1, f"{model.names[cls_id]} {score:.2f}"))

                if labels:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    for x1, y1, label in labels:
                        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
                        # 目标贴近画面顶部时把标签放到框内，防止文字被裁掉
                        top = y1 - text_h if y1 >= text_h else y1
                        draw.rectangle([x1, top, x1 + text_w, top + text_h], fill=(0, 255, 0))
                        draw.text((x1, top), label, font=font, fill=(0, 0, 0))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                writer.write(frame)

            print("✅ 视频处理完成")

        except Exception as e:
            print(f"❌ 视频处理错误: {e}")
            raise
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on a video file or camera source.')
    parser.add_argument('model', help='模型权重路径，例如 runs/detect/train_result/weights/best.pt')
    parser.add_argument('source', help='视频路径或摄像头编号，例如 0')
    parser.add_argument('--output', default='output.mp4', help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    args = parser.parse_args()

    try:
        VideoPredictor.run(args.model, args.source, args.output, conf=args.conf)
    except Exception:
        parser.exit(1)

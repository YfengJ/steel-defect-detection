import argparse


class VideoPredictor:
    @staticmethod
    def run(model_path, source, output_path="output.mp4"):
        """
        独立运行的视频预测函数
        """
        cap = None
        writer = None
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            model = YOLO(model_path)

            # 打开视频
            cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频源: {source}")

            # 获取属性
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            # 输出设置
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise ValueError(f"无法创建输出视频: {output_path}")

            # 字体加载（带容错）
            try:
                font = ImageFont.truetype("simhei.ttf", 20)
            except IOError:
                print("⚠️ 未找到 simhei.ttf，使用系统默认字体")
                font = ImageFont.load_default()

            print(f"▶ 开始处理视频，保存至: {output_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 推理
                results = model(frame, conf=0.25, verbose=False)

                # 手动绘制以支持中文标签：先用 OpenCV 画所有框，
                # 再整帧转一次 PIL 统一绘制文字，避免每个目标都做一次昂贵的色彩空间转换
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{model.names[cls_id]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        detections.append((x1, y1, label))

                if detections:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    for x1, y1, label in detections:
                        # 文字背景
                        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
                        draw.rectangle(
                            [x1, y1 - text_h, x1 + text_w, y1], fill=(0, 255, 0)
                        )
                        draw.text((x1, y1 - text_h), label, font=font, fill=(0, 0, 0))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                writer.write(frame)

            print("✅ 视频处理完成")
            return True

        except Exception as e:
            print(f"❌ 视频处理错误: {e}")
            return False
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on a video file or camera source."
    )
    parser.add_argument(
        "model", help="模型权重路径，例如 runs/detect/train_result/weights/best.pt"
    )
    parser.add_argument("source", help="视频路径或摄像头编号，例如 0")
    parser.add_argument("--output", default="output.mp4", help="输出视频路径")
    args = parser.parse_args()

    from path_validation import validate_model_path, validate_video_source

    for error in (validate_model_path(args.model), validate_video_source(args.source)):
        if error:
            print(f"❌ {error}")
            return 2
    return 0 if VideoPredictor.run(args.model, args.source, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class VideoPredictor:
    @staticmethod
    def run(model_path, source, output_path="output.mp4"):
        """
        独立运行的视频预测函数
        """
        try:
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
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            # 字体加载（带容错）
            try:
                font = ImageFont.truetype("simhei.ttf", 20)
            except IOError:
                print("⚠️ 未找到 simhei.ttf，使用系统默认字体")
                font = ImageFont.load_default()

            print(f"▶ 开始处理视频，保存至: {output_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 推理
                results = model(frame, conf=0.25, verbose=False)

                # 绘制 (使用YOLO自带的plot更方便，或者手动绘制)
                # 这里演示手动绘制以支持中文
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # 坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # 类别与置信度
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{model.names[cls_id]} {conf:.2f}"

                        # CV2 画框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # PIL 画中文
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        # 文字背景
                        text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
                        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=(0, 255, 0))
                        draw.text((x1, y1 - text_h), label, font=font, fill=(0, 0, 0))

                        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                writer.write(frame)

            cap.release()
            writer.release()
            print("✅ 视频处理完成")

        except Exception as e:
            print(f"❌ 视频处理错误: {e}")


if __name__ == "__main__":
    # 简单的测试入口
    import sys

    if len(sys.argv) > 2:
        VideoPredictor.run(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python video_predict.py <model.pt> <video.mp4>")
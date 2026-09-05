# -*- coding: utf-8 -*-
"""نسخه تشخیصی - فقط برای پیدا کردن گلوگاه کندی"""

import os
import sys
import time
import cv2
import numpy as np
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGE_PROTO = os.path.join(BASE_DIR, "models", "age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "models", "age_net.caffemodel")
FONT_PATH = os.path.join(BASE_DIR, "fonts", "Vazirmatn-Regular.ttf")

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

STREAM_URL = "http://172.19.7.81:8080/video"
ROTATE_FRAME = cv2.ROTATE_90_CLOCKWISE
DETECT_EVERY_N_FRAMES = 10


def load_models():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    return face_cascade, age_net


def get_persian_font(size=28):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_persian_text(frame_bgr, text, position, font, color=(0, 255, 0)):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = position
    draw.rectangle([x - 6, y - 4, x + text_w + 6, y + text_h + 10], fill=(0, 0, 0, 160))
    draw.text((x, y), bidi_text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def estimate_age(face_img, age_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN, swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    idx = preds[0].argmax()
    return AGE_BUCKETS[idx], float(preds[0][idx])


class MjpegStream:

    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    def __init__(self, url, timeout=5):
        self.url = url
        self.timeout = timeout
        self.buf = b""
        self.resp = urllib.request.urlopen(url, timeout=timeout)

    def read(self):
        start = self.buf.find(self.SOI)
        end = self.buf.find(self.EOI)
        if start != -1 and end != -1 and end > start:
            jpg = self.buf[start:end + 2]
            self.buf = self.buf[end + 2:]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        try:
            chunk = self.resp.read(65536)
        except Exception as e:
            print("اتصال قطع شد، تلاش دوباره:", e)
            time.sleep(0.5)
            try:
                self.resp = urllib.request.urlopen(self.url, timeout=self.timeout)
                self.buf = b""
            except Exception:
                pass
            return None

        if not chunk:
            return None

        self.buf += chunk
        if len(self.buf) > 2_000_000:
            self.buf = b""
        return None

def main():
    face_cascade, age_net = load_models()
    font = get_persian_font(26)
    stream = MjpegStream(STREAM_URL)

    print("برنامه اجرا شد. برای خروج کلید q را بزن.")

    frame_count = 0
    last_results = []

    t_net_total = t_detect_total = t_draw_total = t_show_total = 0.0
    n_frames_measured = 0
    fps_timer = time.time()
    fps_counter = 0

    while True:
        t0 = time.time()
        frame = stream.read()
        t1 = time.time()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if ROTATE_FRAME is not None:
            frame = cv2.rotate(frame, ROTATE_FRAME)

        frame_count += 1
        t2 = time.time()
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(90, 90))
            last_results = []
            for (x, y, w, h) in faces:
                pad = int(0.15 * w)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                age_range, conf = estimate_age(face_img, age_net)
                last_results.append((x, y, w, h, age_range, conf))
        t3 = time.time()

        for (x, y, w, h, age_range, conf) in last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = f"سن تقریبی: {age_range} سال  ({conf*100:.0f}٪)"
            frame = draw_persian_text(frame, label, (x, max(0, y - 40)), font)
        t4 = time.time()

        cv2.imshow("تخمین سن - Age Estimator", frame)
        key = cv2.waitKey(1) & 0xFF
        t5 = time.time()

        t_net_total += (t1 - t0)
        t_detect_total += (t3 - t2)
        t_draw_total += (t4 - t3)
        t_show_total += (t5 - t4)
        n_frames_measured += 1
        fps_counter += 1

        if time.time() - fps_timer >= 1.0:
            print(
                f"FPS:{fps_counter} | شبکه:{t_net_total/n_frames_measured*1000:.1f}ms "
                f"| تشخیص:{t_detect_total/n_frames_measured*1000:.1f}ms "
                f"| رسم‌متن:{t_draw_total/n_frames_measured*1000:.1f}ms "
                f"| نمایش:{t_show_total/n_frames_measured*1000:.1f}ms"
            )
            t_net_total = t_detect_total = t_draw_total = t_show_total = 0.0
            n_frames_measured = 0
            fps_counter = 0
            fps_timer = time.time()

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

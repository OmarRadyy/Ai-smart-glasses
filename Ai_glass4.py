import cv2
import time
import os
import queue
import threading
from datetime import datetime
import random
from ultralytics import YOLO
from gtts import gTTS
import numpy as np
import pygame   # لازم تثبته: pip install pygame

# ================ إعدادات ================
MODEL_PATH = "yolov8n.pt"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FOCAL_LENGTH = 600
PROCESS_EVERY_N_FRAMES = 3
ANNOUNCE_COOLDOWN = 3.0

translations = {
    "person": "شخص", "car": "سيارة", "bicycle": "دراجة",
    "chair": "كرسي", "dog": "كلب", "cat": "قطة", "table": "طاولة","elephant": "فيل"
}

real_height = {
    "person": 1.7, "car": 1.5, "bicycle": 1.2, "chair": 1.0,
    "dog": 0.6, "cat": 0.3, "table": 0.8 , "elephant": 0.9
}

# دي وحده الصوت باستخدام pygame
def start_audio_worker():
    q = queue.Queue()
    stop_flag = threading.Event()
    pygame.mixer.init()  # تهيئة الميكسر

    def worker():
        while not stop_flag.is_set():
            try:
                text = q.get(timeout=0.5)
            except queue.Empty:
                continue

            filename = None
            try:
                # أنشئ ملف مؤقت من gTTS
                filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{random.randint(0,9999)}.mp3"
                gTTS(text=text, lang='ar').save(filename)

                # شغّل الملف باستخدام pygame (غير محظور)
                try:
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()
                    # أثناء التشغيل نراقب stop_flag فنوقف فورًا لو اتطلب
                    while pygame.mixer.music.get_busy():
                        if stop_flag.is_set():
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.05)
                except Exception as e:
                    print("⚠️ خطأ في تشغيل الصوت (pygame):", e)

            except Exception as e:
                print("⚠️ خطأ أثناء تحضير الصوت:", e)
            finally:
                # احذف الملف لو اتعمل
                try:
                    if filename and os.path.exists(filename):
                        os.remove(filename)
                except Exception as e:
                    print("⚠️ خطأ عند حذف الملف الصوتي:", e)
                # دي علامة انتهاء مهمة الصف 
                try:
                    q.task_done()
                except Exception:
                    pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # دالة لوقف العامل فورًا (توقف التشغيل الحالي أيضاً)
    def stop():
        stop_flag.set()
        try:
            # وقف اي صوت شغال دلوقتي
            pygame.mixer.music.stop()
        except Exception:
            pass

    return q, stop

#      دوال المساعدة 
def estimate_distance(obj_name, pixel_height):
    if obj_name not in real_height or pixel_height <= 0:
        return None
    return (real_height[obj_name] * FOCAL_LENGTH) / pixel_height

def detect_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "على اليسار"
    elif x_center > (2 * frame_width) / 3:
        return "على اليمين"
    return "أمامك"

def make_message(name, direction, distance):
    d = round(distance, 1)
    if distance < 0.5:
        return f"{translations[name]} {direction} على بعد نصف متر! احذر!"
    elif distance < 1.0:
        return f"{translations[name]} {direction} على بعد متر تقريبًا!"
    else:
        return f"{translations[name]} {direction} على بعد {d} متر"

# الفانكشن الرئيسي بقي 
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ لم يتم فتح الكاميرا")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    audio_q, stop_audio = start_audio_worker()
    last_time = {}
    frame_count = 0

    print("✅ النظام يعمل... اضغط Q أو ESC للخروج (الصوت سيتوقف فورًا)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # نعرض بدون تحليل لبعض الفريمات لتخفيف الحمل
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.imshow("Smart Glasses", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("⏹️ تم الضغط على Q/ESC - إغلاق...")
                    break
                continue

            results = model(frame)
            annotated = results[0].plot()
            width = frame.shape[1]
            now = time.time()

            boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, "cpu") else np.array(results[0].boxes.xyxy)
            classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes.cls, "cpu") else np.array(results[0].boxes.cls)

            for box, cls in zip(boxes, classes):
                name = results[0].names[int(cls)]
                if name not in translations:
                    continue

                x1, y1, x2, y2 = box
                distance = estimate_distance(name, abs(y2 - y1))
                if not distance or distance > 4.0:
                    continue

                direction = detect_direction((x1 + x2) / 2, width)
                if now - last_time.get(name, 0) < ANNOUNCE_COOLDOWN:
                    continue
                last_time[name] = now

                msg = make_message(name, direction, distance)
                print("🔊", msg)
                try:
                    audio_q.put_nowait(msg)
                except queue.Full:
                    pass

            cv2.imshow("Smart Glasses", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                print("⏹️ تم الضغط على Q/ESC - إغلاق...")
                break

    finally:
        # وقف العامل (هيوقف التشغيل الحالى فورًا)
        stop_audio()
        # نفض الصف لكن من غير انتظار صوت يخلص لأن إحنا بنوقفه فعليًا
        try:
            while not audio_q.empty():
                audio_q.get_nowait()
                audio_q.task_done()
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        print("🔴 انتهى التشغيل")

if __name__ == "__main__":
    main()

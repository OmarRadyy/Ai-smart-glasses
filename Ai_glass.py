import cv2             # OpenCV لمعالجة الفيديو والصور
import torch           # PyTorch لتشغيل موديلات AI
import numpy as np
from ultralytics import YOLO

#  تحميل موديل YOLOv5 جاهز
# هنا بنحمل نسخة صغيرة وخفيفة YOLOv5s
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).  النسخه دي قدمت واتحذفت تقريبا
model = YOLO('yolov5s.pt')
#  ده تهيئة الكاميرا ث
cap = cv2.VideoCapture(0)  # 0 معناها الكاميرا الأساسية

# ودي حلقة معالجة الفيديو 
while True:
    ret, frame = cap.read()       #وده قراءة صورة من الكاميرا 
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # تشغيل الموديل على الصورة
    results = model(rgb_frame)
    frame = results[0].plot()

    # عرض الفيديو
    cv2.imshow('Smart Glasses AI', frame)

    #q الخروج  لما اضغط على 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  ده عشان انظف الموارد 
cap.release()
cv2.destroyAllWindows()

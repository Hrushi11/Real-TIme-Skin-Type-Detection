import cv2
import time
import tensorflow as tf
from Crop_img import getImg
import FaceDetectionModule as ftm

path = "assets/oil-vid.mp4"
class_names = ["Dry Skin", "Oily Skin"]
cap = cv2.VideoCapture(path)
pTime = 0
detector = ftm.faceDetector(minDetectionCon=0.5)

# loading the model
model = tf.keras.models.load_model("Skin-Type-Recognition-Degraded")

# Preprocess img function
IMG_SIZE = (224, 224)
def load_and_prep(filepath):
  img_path = tf.io.read_file(filepath)
  img = tf.io.decode_image(img_path)
  img = tf.image.resize(img, IMG_SIZE)

  return img

filename = getImg()
filepath = f"./RealTimeDetections/{filename}"
img_pred = load_and_prep(filepath)
pred_prob = model.predict(tf.expand_dims(img_pred, axis=0))
pred_class = class_names[pred_prob.argmax()]
title = f"Skin Type: {pred_class}, Prob: {pred_prob.max():.2f}%"
# print(pred_class, pred_prob)

while True:
    ret, img = cap.read()
    if not ret:
        cap = cv2.VideoCapture(path)
        ret, img = cap.read()

    img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
    img, bboxs = detector.findFaces(img)
    # print(detector.model)

    x, y, w, h = bboxs[0][1]
    cv2.putText(img, text=title, org=(x, y+h+22), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                color=(0, 255, 0), thickness=2)

    cTime = time.time()
    if (cTime - pTime) != 0:
        fps = 1 / (cTime - pTime)
        pTime = cTime

    cv2.putText(img, str(int(fps)), org=(15, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                thickness=3, color=(255, 255, 255))
    cv2.imshow("Video", img)
    # cv2.imshow("Crop", img_pred)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


import tensorflow as tf
from Crop_img import getImg
# print(tf.__version__)

model = tf.keras.models.load_model("Skin-Type-Recognition-Degraded")
# print(model.summary())

# Preprocess img function
IMG_SIZE = (224, 224)
def load_and_prep(filepath):
  img_path = tf.io.read_file(filepath)
  img = tf.io.decode_image(img_path)
  img = tf.image.resize(img, IMG_SIZE)

  return img

class_names = ["Dry Skin", "Oily Skin"]
filename = getImg()
filepath = f"./RealTimeDetections/{filename}"
img_pred = load_and_prep(filepath)
pred_prob = model.predict(tf.expand_dims(img_pred, axis=0))
pred_class = class_names[pred_prob.argmax()]

print(pred_class, pred_prob.max())
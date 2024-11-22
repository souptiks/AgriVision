import tensorflow as tf

model = tf.keras.models.load_model("Traning_model.h5")
model.save("newmodel.h5")
# AI-Based-Plant-Disease-Diagnosis-System
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = data.flow_from_directory(
    "dataset/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val = data.flow_from_directory(
    "dataset/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(train.num_classes,activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=val, epochs=8)
model.save("plant_disease_model.h5")

print("Model trained and saved!")

remedies = {
    "Tomato___Leaf_Mold": "Spray fungicide and remove infected leaves.",
    "Tomato___Healthy": "Plant is healthy.",
    "Potato___Early_Blight": "Use copper fungicide and rotate crops.",
    "Pepper___Bacterial_spot": "Use bactericide and disease-free seeds."
}

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from remedies import remedies

app = Flask(__name__)

model = tf.keras.models.load_model("plant_disease_model.h5")
classes = list(remedies.keys())

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    pred = model.predict(img)
    idx = np.argmax(pred)

    return jsonify({
        "Disease": classes[idx],
        "Remedy": remedies[classes[idx]]
    })

app.run(debug=True)

import streamlit as st
import requests
from PIL import Image

st.title("🌿 Plant Disease Diagnosis")

file = st.file_uploader("Upload Leaf Image")

if file:
    st.image(file, use_column_width=True)

    res = requests.post("http://127.0.0.1:5000/predict", files={"file": file})
    data = res.json()

    st.success("Disease: " + data["Disease"])
    st.info("Remedy: " + data["Remedy"])

tensorflow
numpy
pillow
flask
streamlit
matplotlib

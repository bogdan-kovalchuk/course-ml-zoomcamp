# -*- coding: utf-8 -*-
import json
from io import BytesIO
from urllib import request

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "hair_classifier_empty.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")

def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def preprocess_image(img: Image.Image, target_size=(200, 200)) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)

    x = np.array(img).astype("float32") / 255.0
    x = (x - MEAN) / STD 
    x = np.transpose(x, (2, 0, 1))  
    x = np.expand_dims(x, 0)
    return x


session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name


def predict(url: str) -> float:
    img = download_image(url)
    x = preprocess_image(img)
    output = session.run(None, {input_name: x})[0]
    prediction = float(output[0][0])
    return prediction


def lambda_handler(event, context):
    if isinstance(event, str):
        event = json.loads(event)

    url = event.get("url")
    if not url:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Field 'url' is required"})
        }

    pred = predict(url)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": pred,
            "rounded": round(pred, 2)
        })
    }


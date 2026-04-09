from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import json
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def normalize_config(obj):
    """
    Recursively normalize a Keras 3.x model config so Keras 2.x can load it.
    Handles:
      - quantization_config, optional, module, registered_name  → strip
      - dtype as DTypePolicy dict                               → flatten to string
      - batch_shape                                             → batch_input_shape
    """
    if isinstance(obj, list):
        return [normalize_config(i) for i in obj]

    if not isinstance(obj, dict):
        return obj

    # Flatten dtype: {'class_name': 'DTypePolicy', 'config': {'name': 'float32'}} → 'float32'
    if obj.get("class_name") == "DTypePolicy":
        return obj.get("config", {}).get("name", "float32")

    STRIP = {"quantization_config", "optional", "module", "registered_name"}
    result = {}
    for k, v in obj.items():
        if k in STRIP:
            continue
        result[k] = normalize_config(v)

    # Fix InputLayer batch_shape → batch_input_shape
    if result.get("class_name") == "InputLayer":
        cfg = result.get("config", {})
        if "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")

    return result


def load_model_safe(path):
    with h5py.File(path, "r") as f:
        raw_config = json.loads(f.attrs["model_config"])

    clean_config = normalize_config(raw_config)
    model = tf.keras.models.model_from_json(json.dumps(clean_config))
    model.load_weights(path)
    return model


model = load_model_safe("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
maxlen = 100
label_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}


def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded, verbose=0)
    return label_map[int(np.argmax(pred))]


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["text"]
        result = predict(text)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
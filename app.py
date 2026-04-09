from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import json
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Known bad keys introduced in Keras 3.x that Keras 2.x doesn't understand
UNKNOWN_KEYS = {"optional", "quantization_config", "dtype"}

def fix_config(cfg):
    if isinstance(cfg, dict):
        for key in UNKNOWN_KEYS:
            cfg.pop(key, None)
        if "batch_shape" in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
        # Simplify nested initializer/regularizer dicts that use Keras 3 format
        for k, v in list(cfg.items()):
            if isinstance(v, dict) and "class_name" in v and "module" in v:
                v.pop("module", None)
                v.pop("registered_name", None)
            fix_config(v)
    elif isinstance(cfg, list):
        for item in cfg:
            fix_config(item)

def load_model_compat(path):
    with h5py.File(path, "r") as f:
        model_config = json.loads(f.attrs["model_config"])
    fix_config(model_config)
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    model.load_weights(path)
    return model

model = load_model_compat("model.h5")

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
    pred = model.predict(padded)
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
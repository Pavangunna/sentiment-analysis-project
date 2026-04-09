from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import json
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_model_safe(path):
    """
    Keras 3.x saves quantization_config in layer configs which older and newer
    Keras both reject. This strips all unknown keys before deserializing, then
    loads the weights separately.
    """
    STRIP_KEYS = {"quantization_config", "optional", "module", "registered_name"}

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k not in STRIP_KEYS}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    with h5py.File(path, "r") as f:
        raw_config = json.loads(f.attrs["model_config"])

    clean_config = clean(raw_config)

    # Fix batch_shape → batch_input_shape for any InputLayer
    def fix_input(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "InputLayer" and "batch_shape" in obj.get("config", {}):
                obj["config"]["batch_input_shape"] = obj["config"].pop("batch_shape")
            for v in obj.values():
                fix_input(v)
        elif isinstance(obj, list):
            for i in obj:
                fix_input(i)

    fix_input(clean_config)

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
from PIL import Image
from io import BytesIO
import pickle
import traceback
from reward_server.deqa import load_deqascore
import numpy as np
import os

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_deqascore()

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        # expects a dict with "images", "queries", and optionally "answers"
        # images: (batch_size,) of JPEG bytes
        # queries: (batch_size, num_queries_per_image) of strings
        # answers: (batch_size, num_queries_per_image) of strings
        data = pickle.loads(data)

        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]

        print(f"Got {len(images)} images")

        outputs = INFERENCE_FN(images)

        response = {"outputs": outputs}

        # returns: a dict with "outputs" and optionally "scores"
        # outputs: (batch_size, num_queries_per_image) of strings
        # precision: (batch_size, num_queries_per_image) of floats
        # recall: (batch_size, num_queries_per_image) of floats
        # f1: (batch_size, num_queries_per_image) of floats
        response = pickle.dumps(response)

        returncode = 200
    except Exception as e:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


HOST = "127.0.0.1"
PORT = 8085

if __name__ == "__main__":
    create_app().run(HOST, PORT)
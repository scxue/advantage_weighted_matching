import requests
from PIL import Image
import io
import pickle
import glob
import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

BATCH_SIZE = 24

paths = [os.path.join(os.getcwd(), "a photo of a brown giraffe and a white stop sign.png")]

def f(_):
    for i in tqdm.tqdm(range(0, len(paths), BATCH_SIZE), desc="Processing batches", unit="batch", ncols=100, leave=True, dynamic_ncols=True):
        batch_paths = paths[i : i + BATCH_SIZE]

        jpeg_data = []
        queries = []
        answers = []
        for path in batch_paths:
            image = Image.open(path)

            # Compress the images using JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            jpeg_data.append(buffer.getvalue())

        data = {
            "images": jpeg_data,
            "meta_datas": [{"tag": "color_attr", "include": [{"class": "giraffe", "count": 1, "color": "brown"}, {"class": "stop sign", "count": 1, "color": "white"}], "prompt": "a photo of a brown giraffe and a white stop sign"}],
            "only_strict": False,
        }
        
        data_bytes = pickle.dumps(data)

        # Send the JPEG data in an HTTP POST request to the server
        url = "http://127.0.0.1:18085"
        response = requests.post(url, data=data_bytes)

        # Print the response from the server
        response_data = pickle.loads(response.content)
        # for output in response_data["outputs"]:
        #     print(output)
        #     print("--")

# with ThreadPoolExecutor(max_workers=8) as executor:
#     for _ in executor.map(f, range(8)):
#         pass
f(1)
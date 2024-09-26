import cv2
import requests
import uuid
import numpy as np

# NOTE: posting to faceswap

# session_id = "2f85c33f-c1b3-4ef5-9928-4b9ab8bca68e"
# url = f"http://localhost:8001/api/ai/faceswap?session_id={session_id}&gender=male&template=film"
# image_path = "./tmp/test.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# _, enc_res = cv2.imencode(".jpg", image)
# data = enc_res.tobytes()
#
# r = requests.post(
# 	url=url,
# 	data=data,
# 	headers={"Content-Type": "image/jpeg"},
# )
#
# print(r)


# NOTE: fetching data

url = "http://localhost:8001/api/debug_data/3"
res = requests.get(url)
print(res.status_code)
print(res.json())

# NOTE: image

url = "http://localhost:8001/api/debug_image/3"
res = requests.get(url)
print(res.status_code)
print(res.headers["Content-Type"])
img_buf = np.frombuffer(res.content, np.uint8)
print(img_buf.size)
img = cv2.imdecode(img_buf, -1)


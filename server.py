import os
import time
import logging
import asyncio
import socket
import json
import requests

from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

logging.basicConfig()
logger = logging.getLogger("faceswap")
logger.setLevel(logging.INFO)

MOBIUS_SERVER = os.getenv("MOBIUS_SERVER", "http://localhost:8001")
HOSTNAME = socket.gethostname()


def run_socket(socket):
	message = socket.recv()
	logger.info(f"got: {message}")
	# TODO: whats in the message?
	data = json.loads(message)
	# fetch metadata from the server
	id = data['id']
	url = f"{MOBIUS_SERVER}/api/debug_data/{id}"
	# contains
	#  - session_id
	#  - gender
	#  - template
	res = requests.get(url)
	if res.status_code != 200:
		logger.error(f"Error fetching debug data from server {res.status_code} - {res.text})
		continue
	meta = res.json()
	session_id = meta["session_id"]
	# fetch image from server
	url = f"{MOBIUS_SERVER}/api/debug_image/{id}"
	res = requests.get(url)
	if res.status_code != 200:
		logger.error(f"Error fetching image from server {res.status_code} - {res.text})
		continue
	img_buf = np.frombuffer(res.content, np.uint8)
	img = cv2.imdecode(img_buf, -1)

	# TODO: add model and face swap

	# post result
	url = f"{MOBIUS_FRAME}/api/session/{session_id}/blob"
	r = requests.post(
		url=url,
		data=resp_bytes,
		headers={"Content-Type": "image/jpeg"},
	)
	logger.info(f"{url} returned code:{r.status_code}, content: {r.content}")
	socket.send("OK");


def ws_connect():
	logger.info("Connecting to ws server: {MOBIUS_SERVER}...")
	if "https" in MOBIUS_SERVER:
		ws_url = MOBIUS_SERVER.replace("https", "wss")
	else:
		ws_url = MOBIUS_SERVER.replace("http", "ws")
	while True:
		try:
			with connect(f"{MOBIUS_SERVER}/api/ai/register?hostname={HOSTNAME}") as socket:
				logger.info(f"Connected to ws server: {MOBIUS_SERVER}")
				while True:
					run_socket(socket)
		except ConnectionClosed:
			logger.error("ws connection closed... retry")


ws_connect()


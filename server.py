import os
import time
import logging
import asyncio
import socket
import json
import requests
import sys
import cv2

import numpy as np

from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

import facefusion.choices
import facefusion.globals
from facefusion import (content_analyser, core, face_analyser, face_masker,
                        vision)
from facefusion.content_analyser import pre_check
from facefusion.memory import limit_system_memory
from facefusion.normalizer import normalize_padding
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.core import get_frame_processors_modules

logging.basicConfig()
logger = logging.getLogger("faceswap")
logger.setLevel(logging.INFO)

MOBIUS_SERVER = os.getenv("MOBIUS_SERVER", "http://localhost:8001")
HOSTNAME = socket.gethostname()


def add_overlay(background, overlay):
	alpha_channel = overlay[:, :, 3]
	overlay_rgb = overlay[:, :, :3]
	mask = alpha_channel / 255.0
	inv_mask = 1.0 - mask
	background_rgb = cv2.resize(
	    background,
	    (overlay_rgb.shape[1] - 40, overlay_rgb.shape[0] - 40),
	    interpolation=cv2.INTER_LINEAR,
	)
	background_rgb = cv2.copyMakeBorder(
	    background_rgb, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=[0, 0, 0, 0]
	)
	blended_rgb = overlay_rgb * mask[..., None] + background_rgb * inv_mask[..., None]
	final_image = np.concatenate((blended_rgb, background_rgb[:, :, 3:]), axis=2)
	return final_image

def run_model(session_id, img):
	target_frame = cv2.imread("/home/marin/Downloads/Guinnes/Templates-20240315T081037Z-001/Templates/Male/00000-428648390.jpg")
	print(f"Stat process image {session_id}")
	result_image = core.v2_process_image([img], target_frame)
	print(f"End process image {session_id}")
	return result_image

def run_socket(socket):
	while True:
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
			logger.error(f"Error fetching debug data from server {res.status_code} - {res.text}")
			continue
		meta = res.json()
		session_id = meta["session_id"]
		# fetch image from server
		url = f"{MOBIUS_SERVER}/api/debug_image/{id}"
		res = requests.get(url)
		if res.status_code != 200:
			logger.error(f"Error fetching image from server {res.status_code} - {res.text}")
			continue
		img_buf = np.frombuffer(res.content, np.uint8)
		img = cv2.imdecode(img_buf, -1)

		# face swap
		final_image = run_model(session_id, img)

		# TODO: add overlay

		_, enc_res = cv2.imencode(".jpg", final_image)
		resp_bytes = enc_res.tobytes()

		# post result
		url = f"{MOBIUS_SERVER}/api/session/{session_id}/blob"
		r = requests.post(
			url=url,
			data=resp_bytes,
			headers={"Content-Type": "image/jpeg"},
		)
		logger.info(f"{url} returned code:{r.status_code}, content: {r.content}")
		socket.send("OK");


def ws_connect():
	if "https" in MOBIUS_SERVER:
		ws_url = MOBIUS_SERVER.replace("https", "wss")
	else:
		ws_url = MOBIUS_SERVER.replace("http", "ws")
	logger.info(f"Connecting to ws server: {ws_url}...")
	while True:
		try:
			with connect(f"{ws_url}/api/ai/register?hostname={HOSTNAME}") as socket:
				logger.info(f"Connected to ws server: {ws_url}")
				run_socket(socket)
		except ConnectionClosed:
			logger.error("ws connection closed... retry")


def apply_args():
	# misc
	facefusion.globals.skip_download = os.environ.get("SKIP_DOWNLOAD", False)
	facefusion.globals.headless = True
	facefusion.globals.log_level = "info"
	# execution
	facefusion.globals.execution_providers = [
	    "CUDAExecutionProvider",
	]
	facefusion.globals.execution_thread_count = 4
	facefusion.globals.execution_queue_count = 1
	# memory
	facefusion.globals.video_memory_strategy = "tolerant"
	facefusion.globals.system_memory_limit = 12
	# face analyser
	facefusion.globals.face_analyser_order = "large-small"
	facefusion.globals.face_analyser_age = None
	facefusion.globals.face_analyser_gender = None
	facefusion.globals.face_detector_model = "yoloface"
	facefusion.globals.face_recognizer_model = "arcface_inswapper"
	facefusion.globals.face_detector_size = "640x640"
	facefusion.globals.face_detector_score = 0.5

	# face selector
	facefusion.globals.face_selector_mode = "reference"
	facefusion.globals.reference_face_position = 0
	facefusion.globals.reference_face_distance = 0.6
	facefusion.globals.reference_frame_number = 0

	# face mask
	facefusion.globals.face_mask_types = "box"
	facefusion.globals.face_mask_blur = 0.3
	facefusion.globals.face_mask_padding = normalize_padding([0, 0, 0, 0])
	facefusion.globals.face_mask_regions = facefusion.choices.face_mask_regions
	# frame extraction
	facefusion.globals.trim_frame_start = None
	facefusion.globals.trim_frame_end = None
	facefusion.globals.temp_frame_format = "jpg"
	facefusion.globals.temp_frame_quality = 80
	facefusion.globals.keep_temp = False
	# output creation
	facefusion.globals.output_image_quality = 100
	facefusion.globals.output_video_encoder = "libx264"
	facefusion.globals.output_video_preset = "veryfast"
	facefusion.globals.output_video_quality = 100
	facefusion.globals.skip_audio = True
	# frame processors
	facefusion.globals.frame_processors = ["face_swapper", "face_enhancer"]
	frame_processors_globals.face_swapper_model = "inswapper_128"
	facefusion.globals.face_recognizer_model = "arcface_inswapper"
	frame_processors_globals.face_enhancer_model = "gfpgan_1.4"
	frame_processors_globals.face_enhancer_blend = 80


logger.info("Bootstrap ML models")
apply_args()

if facefusion.globals.system_memory_limit > 0:
	limit_system_memory(facefusion.globals.system_memory_limit)
if (
	not pre_check()
	or not content_analyser.pre_check()
	or not face_analyser.pre_check()
	or not face_masker.pre_check()
):
	logger.error("Error initializing the model")
	sys.exit(1)

for frame_processor_module in get_frame_processors_modules(
    facefusion.globals.frame_processors
):
	if not frame_processor_module.pre_check():
		logger.error(f"Error initializing frace processor {frame_processor_module}")
		sys.exit(1)

ws_connect()


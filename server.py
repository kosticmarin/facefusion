import os
import time
import logging
import asyncio
import socket
import json
import requests

from websockets.sync.client import connect

logging.basicConfig()
logger = logging.getLogger("faceswap")
logger.setLevel(logging.INFO)

MOBIUS_SERVER = os.getenv("MOBIUS_SERVER", "ws://localhost:8001")
HOSTNAME = socket.gethostname()


def ws_connect():
	logger.info("Connecting to ws server: {MOBIUS_SERVER}...")
	with connect(f"{MOBIUS_SERVER}/api/ai/register?hostname={HOSTNAME}") as socket:
		logger.info(f"Connected to ws server: {MOBIUS_SERVER}")
		while(True):
			message = socket.recv()
			logger.info(f"got: {message}")
			data = json.loads(message)
			# TODO: fetch image from server
			# TODO: add model and face swap


ws_connect()


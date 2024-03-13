import logging
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request, Response

import facefusion.choices
import facefusion.globals
from facefusion import (content_analyser, core, face_analyser, face_masker,
                        vision)
from facefusion.content_analyser import pre_check
from facefusion.memory import limit_system_memory
from facefusion.normalizer import normalize_padding
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.core import get_frame_processors_modules

logger = logging.getLogger(__name__)

ROOT_DIR = os.environ.get("ROOT_DIR", "/home/marin")
TARGETS_DIR = f"{ROOT_DIR}/FakeIt/Stable Diffusion images/"
GENDER_DIRS = os.listdir(TARGETS_DIR)
MALE_TARGETS = [
    os.path.join(TARGETS_DIR, GENDER_DIRS[0], fname)
    for fname in os.listdir(os.path.join(TARGETS_DIR, GENDER_DIRS[0]))
]
FEMALE_TARGETS = [
    os.path.join(TARGETS_DIR, GENDER_DIRS[1], fname)
    for fname in os.listdir(os.path.join(TARGETS_DIR, GENDER_DIRS[1]))
]

TEMPLATES_DIR = f"{ROOT_DIR}/FakeIt/Naslovnice"
TEMPLATES = [os.path.join(TEMPLATES_DIR, fname) for fname in os.listdir(TEMPLATES_DIR)]
MALE_COUNTER = 0
FEMALE_COUNTER = 0

WORK_DIR = "./tmp"
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)


def apply_args():
    # misc
    facefusion.globals.skip_download = False
    facefusion.globals.headless = True
    facefusion.globals.log_level = "info"
    # execution
    facefusion.globals.execution_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    facefusion.globals.execution_thread_count = 4
    facefusion.globals.execution_queue_count = 1
    # memory
    facefusion.globals.video_memory_strategy = "strict"
    facefusion.globals.system_memory_limit = 0
    # face analyser
    facefusion.globals.face_analyser_order = "left-right"
    facefusion.globals.face_analyser_age = facefusion.choices.face_analyser_ages
    facefusion.globals.face_analyser_gender = facefusion.choices.face_analyser_genders
    facefusion.globals.face_detector_model = "yoloface"
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
    facefusion.globals.trim_frame_start = 0  # TODO: maybe check
    facefusion.globals.trim_frame_end = 0  # TODO: maybe check
    facefusion.globals.temp_frame_format = "jpg"
    facefusion.globals.temp_frame_quality = 100
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML model
    print("Bootstrap ML models")
    apply_args()

    if facefusion.globals.system_memory_limit > 0:
        limit_system_memory(facefusion.globals.system_memory_limit)
    if (
        not pre_check()
        or not content_analyser.pre_check()
        or not face_analyser.pre_check()
        or not face_masker.pre_check()
    ):
        return
    for frame_processor_module in get_frame_processors_modules(
        facefusion.globals.frame_processors
    ):
        if not frame_processor_module.pre_check():
            return

    yield
    # Clear ML model


app = FastAPI(lifespan=lifespan)


def add_overlay(background, overlay):
    alpha_channel = overlay[:, :, 3]
    overlay_rgb = overlay[:, :, :3]
    mask = alpha_channel / 255.0
    inv_mask = 1.0 - mask
    background_rgb = background[:, :, :3]
    blended_rgb = overlay_rgb * mask[..., None] + background_rgb * inv_mask[..., None]
    final_image = np.concatenate((blended_rgb, background[:, :, 3:]), axis=2)
    return final_image


@app.post("/")
async def process_image(request: Request, gender: str, session_id: str):
    global MALE_COUNTER
    global FEMALE_COUNTER
    img_bytes = await request.body()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, flags=cv2.IMREAD_UNCHANGED)

    image_path = os.path.join(WORK_DIR, f"{session_id}.jpg")
    vision.write_image(image_path, frame=img)
    facefusion.globals.source_paths = [image_path]
    facefusion.globals.output_path = os.path.join(WORK_DIR, f"{session_id}.jpg")

    if gender == "male":
        MALE_COUNTER += 1
        rand_target = MALE_COUNTER % len(MALE_TARGETS)
        facefusion.globals.target_path = MALE_TARGETS[rand_target]
    elif gender == "female":
        FEMALE_COUNTER += 1
        rand_target = FEMALE_COUNTER % len(FEMALE_TARGETS)
        facefusion.globals.target_path = FEMALE_TARGETS[rand_target]
    else:
        raise HTTPException(
            status_code=400, detail="Gender query parameter can be 'male' or 'female'"
        )

    start_time = time.time()
    core.process_image(start_time)

    result_image = vision.read_image(facefusion.globals.output_path)
    rand_overlay = random.randint(0, len(TEMPLATES) - 1)
    print(rand_overlay)
    print(TEMPLATES[rand_overlay])
    overlay = cv2.imread(TEMPLATES[rand_overlay], cv2.IMREAD_UNCHANGED)
    final_image = add_overlay(result_image, overlay)
    _, enc_res = cv2.imencode(".jpg", final_image)
    resp_bytes = enc_res.tobytes()

    # url = f"https://apps.mobiusframe.com/api/session/{session_id}/blob"
    # r = requests.post(
    #     url=url,
    #     data=resp_bytes,
    #     headers={"Content-Type": "image/jpeg"},
    # )
    # print(f"{url} returned code:{r.status_code}, content: {r.content}")
    return Response(content=resp_bytes, media_type="image/jpeg", status_code=200)

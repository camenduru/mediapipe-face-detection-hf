#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import tarfile

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call(shlex.split('pip uninstall -y opencv-python'))
    subprocess.call(shlex.split('pip uninstall -y opencv-python-headless'))
    subprocess.call(
        shlex.split('pip install opencv-python-headless==4.5.5.64'))

import gradio as gr
import huggingface_hub
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

TITLE = 'MediaPipe Face Detection'
DESCRIPTION = 'https://google.github.io/mediapipe/'

HF_TOKEN = os.getenv('HF_TOKEN')


def load_sample_images() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        image_dir.mkdir()
        dataset_repo = 'hysts/input-images'
        filenames = ['001.tar', '005.tar']
        for name in filenames:
            path = huggingface_hub.hf_hub_download(dataset_repo,
                                                   name,
                                                   repo_type='dataset',
                                                   use_auth_token=HF_TOKEN)
            with tarfile.open(path) as f:
                f.extractall(image_dir.as_posix())
    return sorted(image_dir.rglob('*.jpg'))


def run(image: np.ndarray, model_selection: int,
        min_detection_confidence: float) -> np.ndarray:
    with mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
    ) as face_detection:
        results = face_detection.process(image)

    res = image[:, :, ::-1].copy()
    if results.detections is not None:
        for detection in results.detections:
            mp_drawing.draw_detection(res, detection)
    return res[:, :, ::-1]


model_types = [
    'Short-range model (best for faces within 2 meters)',
    'Full-range model (best for faces within 5 meters)',
]

image_paths = load_sample_images()
examples = [[path.as_posix(), model_types[0], 0.5] for path in image_paths]

gr.Interface(
    fn=run,
    inputs=[
        gr.Image(label='Input', type='numpy'),
        gr.Radio(label='Model',
                 choices=model_types,
                 type='index',
                 value=model_types[0]),
        gr.Slider(label='Minimum Detection Confidence',
                  minimum=0,
                  maximum=1,
                  step=0.05,
                  value=0.5),
    ],
    outputs=gr.Image(label='Output', type='numpy'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).launch(show_api=False)

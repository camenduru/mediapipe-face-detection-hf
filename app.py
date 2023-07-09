#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib

import gradio as gr
import huggingface_hub
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

TITLE = 'MediaPipe Face Detection'
DESCRIPTION = 'https://google.github.io/mediapipe/'


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

image_paths = sorted(pathlib.Path('images').rglob('*.jpg'))
examples = [[path, model_types[0], 0.5] for path in image_paths]

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
    outputs=gr.Image(label='Output'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()

#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tarfile

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())

import gradio as gr
import huggingface_hub
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

TITLE = 'MediaPipe Face Detection'
DESCRIPTION = 'https://google.github.io/mediapipe/'
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.mediapipe-face-detection" alt="visitor badge"/></center>'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


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
                                                   use_auth_token=TOKEN)
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


def main():
    gr.close_all()

    args = parse_args()

    model_types = [
        'Short-range model (best for faces within 2 meters)',
        'Full-range model (best for faces within 5 meters)',
    ]

    image_paths = load_sample_images()
    examples = [[path.as_posix(), model_types[0], 0.5] for path in image_paths]

    gr.Interface(
        run,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(model_types,
                            type='index',
                            default=model_types[0],
                            label='Model'),
            gr.inputs.Slider(0,
                             1,
                             step=0.05,
                             default=0.5,
                             label='Minimum Detection Confidence'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()

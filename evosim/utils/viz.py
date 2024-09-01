from typing import List

import cv2
import imageio
import numpy as np
from PIL import Image


def save_episode_gif(
    np_frames: List[np.ndarray],
    filepath: str,
    image_h: int = 256,
    image_w: int = 256,
    duration: float = 0.03,
):
    frames = [
        Image.fromarray(frame.astype(np.uint8)).resize(
            (image_h, image_w), Image.NEAREST
        )
        for frame in np_frames
    ]
    imageio.mimsave(filepath, frames, duration=duration)

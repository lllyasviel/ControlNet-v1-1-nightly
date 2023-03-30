import random

import cv2
import numpy as np
from annotator.util import make_noise_disk, img2mask


class ContentShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = 256
        x = make_noise_disk(H, W, 1, F) * float(W - 1)
        y = make_noise_disk(H, W, 1, F) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


class ColorShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = random.randint(64, 384)
        A = make_noise_disk(H, W, 3, F)
        B = make_noise_disk(H, W, 3, F)
        C = (A + B) / 2.0
        A = (C + (A - C) * 3.0).clip(0, 1)
        B = (C + (B - C) * 3.0).clip(0, 1)
        L = img.astype(np.float32) / 255.0
        Y = A * L + B * (1 - L)
        Y -= np.min(Y, axis=(0, 1), keepdims=True)
        Y /= np.maximum(np.max(Y, axis=(0, 1), keepdims=True), 1e-5)
        Y *= 255.0
        return Y.clip(0, 255).astype(np.uint8)


class GrayDetector:
    def __call__(self, img):
        Y = np.mean(img, axis=2)
        Y = np.stack([Y] * 3, axis=2)
        return Y.clip(0, 255).astype(np.uint8)


class Image2MaskShuffleDetector:
    def __init__(self, resolution=(640, 512)):
        self.H, self.W = resolution

    def __call__(self, img):
        m = img2mask(img, self.H, self.W)
        m *= 255.0
        return m.clip(0, 255).astype(np.uint8)

import cv2
import numpy as np
from annotator.util import make_noise_disk


class ShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = 256
        x = make_noise_disk(H, W, 1, F) * float(W - 1)
        y = make_noise_disk(H, W, 1, F) * float(H - 1)
        flow = np.stack([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)

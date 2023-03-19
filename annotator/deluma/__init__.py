import numpy as np


class DelumaDetector:
    def __call__(self, img):
        y = img.astype(np.float32)

        color = y - np.mean(y, axis=2, keepdims=True)
        intensity = np.mean(y - np.max(y, axis=2, keepdims=True), axis=2, keepdims=True)

        intensity -= np.min(intensity)
        intensity /= np.maximum(np.max(intensity), 1e-5)
        intensity *= 255.0

        y = intensity + color
        y = y.clip(0, 255).astype(np.uint8)
        return y

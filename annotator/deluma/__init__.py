import numpy as np


class DelumaDetector:
    def __call__(self, img):
        y = img.astype(np.float32)

        luma = 0.299 * y[:, :, 0] + 0.587 * y[:, :, 1] + 0.114 * y[:, :, 2]
        color = y - luma[:, :, None]
        intensity = np.min(y, axis=2, keepdims=True) - np.max(y, axis=2, keepdims=True)

        intensity -= np.min(intensity)
        intensity /= np.maximum(np.max(intensity), 1e-5)
        intensity *= 255.0

        y = intensity + color
        y = y.clip(0, 255).astype(np.uint8)
        return y

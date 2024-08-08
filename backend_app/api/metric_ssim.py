import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

def calculate_ssim(img1, img2):
    """Calculates the SSIM between two images."""
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    # SSIM 계산을 위해 이미지의 최소 크기에 맞춰 win_size 설정
    min_side = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_side)  # win_size는 최소 크기 7과 이미지 최소 크기의 작은 값을 사용
    ssim_value, _ = compare_ssim(img1, img2, full=True, win_size=win_size, channel_axis=2)
    return ssim_value

def load_image(image_path, target_size=None):
    """Loads an image, resizes it if a target size is provided, and converts it to RGB."""
    image = Image.open(image_path).convert('RGB')
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def compute_ssim(img1_path, img2_path):
    """Computes the SSIM between two images."""
    img1 = load_image(img1_path)
    img2 = load_image(img2_path, target_size=img1.size)
    
    ssim_value = calculate_ssim(img1, img2)
    return ssim_value

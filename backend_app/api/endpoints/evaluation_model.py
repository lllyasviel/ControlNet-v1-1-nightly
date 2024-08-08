import uuid
import cv2
import time
import os

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from backend_app.api.utils_process import process
from backend_app.core.config import settings
from global_model import ml_models
from backend_app.api.metric_fid import compute_fid
from backend_app.api.metric_ssim import compute_ssim
# from backend_app.api.metric_psnr import calculate_psnr

router = APIRouter()

@router.post("/process")
async def process_image(
    input_image: UploadFile = File(...),
    prompt: str = Form(...),
    a_prompt: str = Form('best quality, extremely detailed'),
    n_prompt: str = Form('longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'),
    num_samples: int = Form(1),
    image_resolution: int = Form(512),
    ddim_steps: int = Form(20),
    guess_mode: bool = Form(False),
    strength: float = Form(1.0),
    scale: float = Form(9.0),
    seed: int = Form(-1),
    eta: float = Form(0.0),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200)
):
    unique_prefix = uuid.uuid4().hex

    apply_canny = ml_models["apply_canny"]
    model = ml_models["BASE_MODEL_V10"]
    ddim_sampler = ml_models["ddim_sampler"]

    input_image = np.fromstring(await input_image.read(), np.uint8)
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)

    # input image 저장
    input_image_path = f'./result_images/{unique_prefix}'
    os.makedirs(input_image_path, exist_ok=True)
    input_image_path = f'./result_images/{unique_prefix}/input.png'
    cv2.imwrite(input_image_path, input_image)
    
    # process가 실행되는 시간 측정
    process_start_time = time.time()

    result_images = process(
        apply_canny, model, ddim_sampler,
        input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps,
        guess_mode, strength, scale, seed, eta, low_threshold, high_threshold
    )

    process_end_time = time.time()
    # process가 실행된 시간
    process_time = process_end_time - process_start_time


    output_paths = []
    result_images = result_images[1:]
    # output image 저장
    for i, image in enumerate(result_images):
        # f'./result_images/{unique_prefix}/input.png'
        output_path = f'./result_images/{unique_prefix}/output_{i}.png'
        output_paths.append(output_path)
        cv2.imwrite(output_path, image)

    # 각 metric 계산
    fid_scores = []
    ssim_scores = []
    # psnr_scores = []

    for result_image in output_paths:
        fid_scores.append(compute_fid(input_image_path, result_image))
        ssim_scores.append(compute_ssim(input_image_path, result_image))
        # psnr_scores.append(calculate_psnr(input_image, result_image))

    return {
        'input_image': input_image_path,
        'process_time': process_time,
        'output_paths': output_paths,
        'score_fid': fid_scores,
        # 'score_psnr': psnr_scores,
        'score_ssim': ssim_scores,
    }

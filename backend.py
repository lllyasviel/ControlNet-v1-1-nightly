from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from typing import Dict, Any
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model = None
ddim_sampler = None

def init_model():
    global model
    global ddim_sampler
    model_name = 'control_v11p_sd15_inpaint'
    model = create_model(f'./models/{model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
    model = model.cuda()
    
    ddim_sampler = DDIMSampler(model)


def process(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur,
            update_state_fn):
    assert prompt
    with torch.no_grad():
        input_image = HWC3(input_image_and_mask['image'])
        input_mask = input_image_and_mask['mask']

        img_raw = resize_image(input_image, image_resolution).astype(np.float32)
        H, W, C = img_raw.shape

        if (len(input_mask.shape) >= 3):
            input_mask = input_mask[:, :, 0]
        mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)

        mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)

        detected_map = img_raw.copy()
        detected_map[mask_pixel > 0.5] = - 255.0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        mask = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda()
        mask = torch.stack([mask for _ in range(num_samples)], dim=0)
        mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

        x0 = torch.from_numpy(img_raw.copy()).float().cuda() / 127.0 - 1.0
        x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
        x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

        mask_pixel_batched = mask_pixel[None, :, :, None]
        img_pixel_batched = img_raw.copy()[None]

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        prompts = [prompt + ', ' + a_prompt] * num_samples
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(prompts)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        def update_status_dict(i):
            update_state_fn({"step": i, "num_steps": ddim_steps + 1})
    
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=True, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond, x0=x0, mask=mask,
            callback = update_status_dict
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
        x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

        results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
    return [detected_map.clip(0, 255).astype(np.uint8)] + results

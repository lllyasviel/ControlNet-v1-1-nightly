from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model_name = 'control_v11p_sd15_inpaint'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur):
    with torch.no_grad():
        input_image = HWC3(input_image_and_mask['image'])
        input_mask = input_image_and_mask['mask']

        img_raw = resize_image(input_image, image_resolution).astype(np.float32)
        H, W, C = img_raw.shape

        mask_pixel = cv2.resize(input_mask[:, :, 0], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
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

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
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

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, x0=x0, mask=mask)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
        x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

        results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
    return [detected_map.clip(0, 255).astype(np.uint8)] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Inpaint Mask")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy", tool="sketch")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            mask_blur = gr.Slider(label="Mask Blur", minimum=0.1, maximum=7.0, value=5.0, step=0.01)
            with gr.Accordion("Advanced options", open=False):
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')

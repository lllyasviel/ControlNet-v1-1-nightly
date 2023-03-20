import gradio as gr

from annotator.util import resize_image, HWC3


model_canny = None


def canny(img, res, l, h):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return [result]


model_hed = None


def hed(img, res):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return [result]


model_pidi = None


def pidi(img, res):
    img = resize_image(HWC3(img), res)
    global model_pidi
    if model_pidi is None:
        from annotator.pidinet import PidiNetDetector
        model_pidi = PidiNetDetector()
    result = model_pidi(img)
    return [result]


model_mlsd = None


def mlsd(img, res, thr_v, thr_d):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return [result]


model_midas = None


def midas(img, res):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector()
    result = model_midas(img)
    return [result]


model_zoe = None


def zoe(img, res):
    img = resize_image(HWC3(img), res)
    global model_zoe
    if model_zoe is None:
        from annotator.zoe import ZoeDetector
        model_zoe = ZoeDetector()
    result = model_zoe(img)
    return [result]


model_normalbae = None


def normalbae(img, res):
    img = resize_image(HWC3(img), res)
    global model_normalbae
    if model_normalbae is None:
        from annotator.normalbae import NormalBaeDetector
        model_normalbae = NormalBaeDetector()
    result = model_normalbae(img)
    return [result]


model_openpose = None


def openpose(img, res, hand_and_face):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result = model_openpose(img, hand_and_face)
    return [result]


model_uniformer = None


def uniformer(img, res):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return [result]


model_lineart_anime = None


def lineart_anime(img, res):
    img = resize_image(HWC3(img), res)
    global model_lineart_anime
    if model_lineart_anime is None:
        from annotator.lineart_anime import LineartAnimeDetector
        model_lineart_anime = LineartAnimeDetector()
    result = model_lineart_anime(img)
    return [result]


model_lineart = None


def lineart(img, res, coarse=False):
    img = resize_image(HWC3(img), res)
    global model_lineart
    if model_lineart is None:
        from annotator.lineart import LineartDetector
        model_lineart = LineartDetector()
    result = model_lineart(img, coarse)
    return [result]


model_oneformer_coco = None


def oneformer_coco(img, res):
    img = resize_image(HWC3(img), res)
    global model_oneformer_coco
    if model_oneformer_coco is None:
        from annotator.oneformer import OneformerCOCODetector
        model_oneformer_coco = OneformerCOCODetector()
    result = model_oneformer_coco(img)
    return [result]


model_oneformer_ade20k = None


def oneformer_ade20k(img, res):
    img = resize_image(HWC3(img), res)
    global model_oneformer_ade20k
    if model_oneformer_ade20k is None:
        from annotator.oneformer import OneformerADE20kDetector
        model_oneformer_ade20k = OneformerADE20kDetector()
    result = model_oneformer_ade20k(img)
    return [result]


model_content_shuffler = None


def content_shuffler(img, res):
    img = resize_image(HWC3(img), res)
    global model_content_shuffler
    if model_content_shuffler is None:
        from annotator.shuffle import ContentShuffleDetector
        model_content_shuffler = ContentShuffleDetector()
    result = model_content_shuffler(img)
    return [result]


model_color_shuffler = None


def color_shuffler(img, res):
    img = resize_image(HWC3(img), res)
    global model_color_shuffler
    if model_color_shuffler is None:
        from annotator.shuffle import ColorShuffleDetector
        model_color_shuffler = ColorShuffleDetector()
    result = model_color_shuffler(img)
    return [result]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Canny Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            low_threshold = gr.Slider(label="low_threshold", minimum=1, maximum=255, value=100, step=1)
            high_threshold = gr.Slider(label="high_threshold", minimum=1, maximum=255, value=200, step=1)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=canny, inputs=[input_image, resolution, low_threshold, high_threshold], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## HED Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=hed, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Pidi Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=pidi, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## MLSD Edge")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            value_threshold = gr.Slider(label="value_threshold", minimum=0.01, maximum=2.0, value=0.1, step=0.01)
            distance_threshold = gr.Slider(label="distance_threshold", minimum=0.01, maximum=20.0, value=0.1, step=0.01)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=mlsd, inputs=[input_image, resolution, value_threshold, distance_threshold], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## MIDAS Depth")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=midas, inputs=[input_image, resolution], outputs=[gallery])


    with gr.Row():
        gr.Markdown("## Zoe Depth")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=zoe, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Normal Bae")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=normalbae, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Openpose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            hand_and_face = gr.Checkbox(label='Hand and Face', value=False)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=openpose, inputs=[input_image, resolution, hand_and_face], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Lineart Anime")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=lineart_anime, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Lineart")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            coarse = gr.Checkbox(label='Using coarse model', value=False)
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=lineart, inputs=[input_image, resolution, coarse], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Uniformer Segmentation")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=uniformer, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Oneformer COCO Segmentation")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=oneformer_coco, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Oneformer ADE20K Segmentation")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=640, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=oneformer_ade20k, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Content Shuffle")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=content_shuffler, inputs=[input_image, resolution], outputs=[gallery])

    with gr.Row():
        gr.Markdown("## Color Shuffle")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            resolution = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button = gr.Button(label="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(height="auto")
    run_button.click(fn=color_shuffler, inputs=[input_image, resolution], outputs=[gallery])


block.launch(server_name='0.0.0.0')

import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import shutil
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-m", "--model_path", type=str, help="path to model", default="dreamlike-art/dreamlike-photoreal-2.0")
parser.add_argument("-i", "--instance_path", type=str, default="out/", help="directory for output images")


def inference(prompt, negative_prompt, num_images):
    negative_prompt_base = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate," \
                      " morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation," \
                      " deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions," \
                      " malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, " \
                      "username, watermark, signature"
    num_images = int(num_images)
    try:
        if os.path.exists(args.instance_path):
            shutil.rmtree(args.instance_path)
        os.makedirs(args.instance_path, exist_ok=True)

        outputs = []
        batch_size = 1
        for i in range(num_images):
            outputs += pipeline("shutterstock poster background photo of " + prompt + " high resoltuion 4k", negative_prompt=
                                negative_prompt_base + negative_prompt, num_inference_steps=25,
                                guidance_scale=7.5, height=512, width=512, num_images_per_prompt=batch_size).images

        filenames = []
        for idx, img in enumerate(outputs):
            filename = os.path.join(args.instance_path, prompt[:50] + f"_{idx}.png")
            img.save(filename)
            filenames.append(os.path.abspath(filename))
    except Exception as e:
        return gr.Gallery.update(value=None), gr.Textbox.update(value='Error: {}'.format(e))

    return gr.Gallery.update(value=filenames), gr.Textbox.update(value='Completed.')


with gr.Blocks() as demo:
    prompt = gr.Textbox(label="prompt")
    neg_prompt = gr.Textbox(label="negative_prompt")
    num_outputs = gr.Number(label="num_outputs")
    output_images = gr.Gallery()
    output_images.style(height="512px")
    running_info = gr.Textbox(label="Log")
    btn_run = gr.Button(value="Run")

    btn_run.click(inference, inputs=[prompt, neg_prompt, num_outputs],
                  outputs=[output_images, running_info])

if __name__ == "__main__":
    args = parser.parse_args()

    pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, safety_checker=None)
    pipeline.to("cuda")

    demo.queue().launch(share=True)

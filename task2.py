import gradio as gr
import shutil
import os
from PIL import Image
from argparse import ArgumentParser
from selenium import webdriver
from time import sleep
import numpy as np
from sklearn.cluster import KMeans

parser = ArgumentParser()
parser.add_argument("-u", "--url", type=str, help="url of webpage")
parser.add_argument("-i", "--instance_path", type=str, default="out/", help="directory for output images")

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800


def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def palette_from_clusters(clusters):
    palette = np.zeros((50, IMAGE_WIDTH, 3), np.uint8)
    steps = IMAGE_WIDTH/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette


def get_pallete(url, num_colors):
    try:
        if os.path.exists(args.instance_path):
            shutil.rmtree(args.instance_path)
        os.makedirs(args.instance_path, exist_ok=True)

        num_colors = int(num_colors)
        pallete_path = f"{args.instance_path}/pallete.png"
        screenshot_path = f"{args.instance_path}/screenshot.png"

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.headless = True
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        sleep(1)
        driver.get_screenshot_as_file(screenshot_path)

        img_ = Image.open(screenshot_path)
        img = Image.new("RGB", img_.size, (255, 255, 255))
        img.paste(img_, mask=img_.split()[3])
        arr = np.asarray(img)
        clt = KMeans(n_clusters=num_colors, random_state=42)

        clt.fit(arr.reshape(-1, 3))
        pallete_img = palette_from_clusters(clt)
        pallete_img = Image.fromarray(np.uint8(pallete_img)).convert('RGB')
        pallete_img = get_concat_v_blank(img, pallete_img)
        pallete_img.save(pallete_path)
    except Exception as e:
        return gr.Image.update(value=None), gr.Textbox.update(value='Error: {}'.format(e))

    return gr.Image.update(value=pallete_path), gr.Textbox.update(value='Completed.')


with gr.Blocks() as demo:
    url = gr.Textbox(label="url")
    num_colors = gr.Number(label="Num colors")
    output_image = gr.Image().style(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    running_info = gr.Textbox(label="Log")
    btn_run = gr.Button(value="run")

    btn_run.click(get_pallete, inputs=[url, num_colors], outputs=[output_image, running_info])


if __name__ == "__main__":
    args = parser.parse_args()
    demo.queue().launch(share=True)

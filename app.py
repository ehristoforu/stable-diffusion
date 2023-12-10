"""
Diffusers WebUI V1.0 with Beta Functionsss
"""

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from transformers import pipeline, set_seed
import gradio as gr, random, re
import torch
from transformers import pipeline, set_seed



gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')
with open("prompts.txt", "r") as f:
    line = f.readlines()

global i2i
i2i = False

def i2i_on(input_image):
    global i2i
    i2i= True
    return input_image
def i2i_off(input_image):
    global  i2i
    i2i = False

def generate(prompt,negative, mode, num_images, mem_use, model, width, height, style, guidance, dreamart_use, input_image):
    global i2i
    if prompt == "" or prompt == " ":
        gr.Warning("We didn't enter a prompt!")
    if dreamart_use:
        seed = random.randint(100, 1000000)
        set_seed(seed)
        response = gpt2_pipe(prompt, max_length=(len(prompt) + random.randint(60, 90)), num_return_sequences=4)
        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != prompt and len(resp) > (len(prompt) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp + '\n')
        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+', '', response_end)
        response_end = response_end.replace("<", "").replace(">", "")
    else:
        response_end = prompt

    if i2i:
        if model == "stabilityai/sd-turbo":
            pipeline = AutoPipelineForImage2Image.from_pretrained(model, use_safetensors=True,
                                                         torch_dtype=torch.float16, low_cpu_mem_usage=mem_use)
        else:
            pipeline = AutoPipelineForImage2Image.from_pretrained(model, use_safetensors=False,
                                                         torch_dtype=torch.float16, low_cpu_mem_usage=mem_use)
    else:
        if model == "stabilityai/sd-turbo":
            pipeline = AutoPipelineForText2Image.from_pretrained(model, use_safetensors=True,
                                                      torch_dtype=torch.float16, low_cpu_mem_usage=mem_use)
        else:
            pipeline = AutoPipelineForText2Image.from_pretrained(model, use_safetensors=False,
                                                     torch_dtype=torch.float16, low_cpu_mem_usage=mem_use)



    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to("cpu")

    images_output = []
    if mode == "Quality":
        steps = 50
    elif mode == "Speed":
        steps = 30
    else:
        steps = 15
    if i2i:
        for _ in range(num_images):
            result = pipeline(
                prompt=f"{response_end}, {style}",
                negative_prompt=f"{negative}",
                image=input_image,
                strength=0.8,
            guidance_scale=guidance
        ).images[0]
            images_output.append(result)

    else:
        for _ in range(num_images):
            result = pipeline(
                prompt=f"{response_end}, {style}",
                negative_prompt=f"{negative}",
                num_inference_steps=steps,
                height=height, width=width,
            guidance_scale=guidance
        ).images[0]
            images_output.append(result)

    return images_output


with gr.Blocks() as demo:
    gr.Markdown("""
    # OpenSkyML's Playground
    """)
    if not torch.cuda.is_available():
        gr.Markdown("CUDA isn't available (not GPU). This demo is not working on CPU!")
    with gr.Row():
        with gr.Column():
            custom_model = gr.Dropdown(label="Custom model",
                                       choices=["openskyml/overall-v1", "stabilityai/sdxl-turbo", "stabilityai/sd-turbo", "runwayml/stable-diffusion-v1-5", "kandinsky-community/kandinsky-2-1", "segmind/SSD-1B", "playgroundai/playground-v2-1024px-base", "stablediffusionapi/deliberate-v3", "stablediffusionapi/juggernaut-xl-v7"],
                                       value="openskyml/overall-v1", interactive=True, allow_custom_value=False)
        with gr.Column():
            mode = gr.Radio(label="Mode", choices=["Quality", "Speed", "Super-speed"], value="Speed", interactive=True)
    with gr.Row():
        gallery = gr.Gallery(show_label=False, columns=6)

    with gr.Row():
        prompt = gr.Textbox(show_label=False, placeholder="Enter your prompt", interactive=True, max_lines=1, scale=3)
        button = gr.Button(value="Run", min_width=5,scale=1)

    with gr.Tab("Settings"):
        with gr.Row():
            with gr.Column():
                negative = gr.Textbox(label="Negative Prompt", placeholder="Enter a negative", interactive=True, max_lines=1)
                style= gr.CheckboxGroup(label="Style", choices=["8K", "Cinematic", "Art", "Sketch", "Anime", "1990s", "1880s", "Watercolor", "Illusion","Illustration"], interactive=True)
            with gr.Column():
                guidance = gr.Slider(label="Guidance Scale", minimum=0.5, maximum=10, step=0.5, value=6, interactive=True)
                images = gr.Slider(label="Number of images", minimum=1, maximum=12, step=1, value=2, interactive=True)
            with gr.Column():
                width = gr.Slider(label="Width", minimum=8, maximum=1024, step=8, value=512, interactive=True)
                height = gr.Slider(label="Height", minimum=8, maximum=1024, step=8, value=512, interactive=True)
    with gr.Tab("Hardware"):
        with gr.Row():
            with gr.Column():
                mem_use = gr.Checkbox(label="Low memory use", value=False, interactive=True)
    with gr.Tab("Input Image"):
        input_image = gr.Image(show_label=False, interactive=True)
    with gr.Tab("Beta"):
        gr.Markdown("""
        ## Beta functions (unstable)
        """)
        with gr.Row():
            with gr.Column():
                dreamart_use = gr.Checkbox(label="DreamART", info="Prompt dreamer (beta)", value=False, interactive=True)



    input_image.upload(i2i_on, inputs=input_image, outputs=input_image)
    input_image.clear(i2i_off, inputs=input_image, outputs=input_image)
    prompt.submit(generate, inputs=[prompt,negative, mode, images, mem_use, custom_model, width, height, style, guidance, dreamart_use, input_image], outputs=gallery)
    button.click(generate, inputs=[prompt, negative, mode, images, mem_use, custom_model, width, height, style,guidance, dreamart_use, input_image], outputs=gallery)

demo.launch(debug=True, share=True)
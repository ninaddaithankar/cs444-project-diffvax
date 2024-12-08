import gradio as gr
import numpy as np
import torch

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


# -- basic hyperparams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "./model-checkpoints/sam/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"


# -- load the segment anything model by Meta AI and init the predictor for getting a mask 
segment_anything_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
segment_anything_model = segment_anything_model.to(DEVICE)

segment_anything_predictor = SamPredictor(segment_anything_model)


# -- load the stable diffusion inpainting pipeline from huggingface
pipe = StableDiffusionInpaintPipeline.from_pretrained(
	"stabilityai/stable-diffusion-2-inpainting",
	torch_dtype = torch.float16,
)
pipe = pipe.to(DEVICE)


# -- gradio demo UI for getting a mask by clicking on a point in the image
selected_pixels = []
with gr.Blocks() as demo:
	with gr.Row():
		input_image = gr.Image(label="Input Image")
		mask_image = gr.Image(label="Mask")
		output_image = gr.Image(label="Output")

	with gr.Row():
		prompt_text = gr.Textbox(lines=1, label="Prompt")

	with gr.Row():
		submit = gr.Button("Submit")

	def generate_mask(image, evt: gr.SelectData):
		selected_pixels.append(evt.index)
		
		segment_anything_predictor.set_image(image=image)
		input_points = np.array(selected_pixels)
		input_labels = np.ones(input_points.shape[0])

		# outputs mask in shape (1, size, size)
		mask, _, _ = segment_anything_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
		mask = Image.fromarray(mask[0, :, :])
		return mask

	def inpaint(image, mask, prompt):
		image = Image.fromarray(image)
		mask = Image.fromarray(mask)

		image = image.resize((512, 512))
		mask = mask.resize((512, 512))

		output = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

		return output

	input_image.select(generate_mask, [input_image], [mask_image])
	submit.click(inpaint, inputs=[input_image, mask_image, prompt_text], outputs=[output_image])


if __name__ == "__main__":
	demo.launch()
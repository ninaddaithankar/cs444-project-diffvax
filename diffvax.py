# implementation for DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing (https://arxiv.org/pdf/2411.17957)

import torch
import torch.optim as optim

from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm

from utils.models import *
from utils.losses import L_noise, L_edit
from utils.dataset import make_image_dataset


# -- hyperparams
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 350
BATCH_SIZE = 4
LR = 1e-5

ALPHA = 4  # weight for L_noise

NUM_WORKERS = 1
TRAIN_DATA_PATHS = []
VAL_DATA_PATHS = []


# -- init the immunizer model (based on UNet++)
immunizer = ImmunizerModel()


# -- load the stable diffusion inpainting pipeline from huggingface
stable_diffusion_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
	"stabilityai/stable-diffusion-2-inpainting",
	torch_dtype = torch.float16,
)
stable_diffusion_pipeline = stable_diffusion_pipeline.to(DEVICE)


# -- transformations
transforms = []


# -- dataloaders
train_dataset, train_loader = make_image_dataset(
	data_paths=TRAIN_DATA_PATHS, 
	batch_size=BATCH_SIZE, 
	num_workers=NUM_WORKERS, 
	pin_mem=True, 
	transform=transforms, 
	shuffle=True)

val_dataset, val_loader = make_image_dataset(
	data_paths=VAL_DATA_PATHS, 
	batch_size=BATCH_SIZE, 
	num_workers=NUM_WORKERS, 
	pin_mem=True, 
	transform=transforms, 
	shuffle=False)



# -- optimization
for param in stable_diffusion_pipeline.parameters():
	param.requires_grad = False
	
optimizer = optim.Adam(immunizer.parameters(), lr=LR)



# -- training loop
for epoch in range(NUM_EPOCHS):
	immunizer.train()
	running_loss = 0.0
	
	progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

	for batch in progress_bar:
		images, masks, prompts = batch
		
		# Generate immunized image
		immunized_image, epsilon_im = immunizer(images, masks)
		
		# Compute the losses
		loss_noise = L_noise(immunized_image, images, masks)
		loss_edit = L_edit(immunized_image, images, masks, stable_diffusion_pipeline, prompts)
		
		# Total loss
		total_loss = ALPHA * loss_noise + loss_edit
		
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		
		running_loss += total_loss.item()
		progress_bar.set_postfix(loss=total_loss.item())
	
	print(f"Epoch {epoch+1} finished. Average Loss: {running_loss / len(train_loader):.4f}")



# -- inference to immunize an image
def immunize_image(image, mask):
	immunizer = ImmunizerModel()

	immunized_image = immunizer(image, mask)

	loss = torch.sum((immunized_image - image) * mask).abs()
	print(f"Imperceptibility Loss: {loss.item()}")

	return immunized_image
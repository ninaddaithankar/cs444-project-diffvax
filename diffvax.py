# implementation for DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing (https://arxiv.org/pdf/2411.17957)

import argparse
import torch
import torch.optim as optim

from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm

from utils.models import *
from utils.losses import L_noise, L_edit
from utils.dataset import CC2_Dataset, get_train_and_val_loaders


def train(args):

	# -- hyperparams
	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

	NUM_EPOCHS = args.epochs				 
	BATCH_SIZE = args.bs								# default: 4
	LR = args.lr										# default: 1e-5

	# weight for the L_noise term
	ALPHA = args.alpha									# default: 4

	NUM_WORKERS = args.num_workers						# default: 1
	DATASET_PATH = args.dataset_path					# default: "/work/hdd/bcsi/ndaithankar/datasets/cc2/"
	TRAIN_SPLIT_PCT = args.train_split_pct				# default: 0.8


	# -- init the immunizer model (based on UNet++)
	immunizer = ImmunizerModel()
	immunizer = immunizer.to(DEVICE)


	# -- load the stable diffusion inpainting pipeline from huggingface
	stable_diffusion_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
		"stabilityai/stable-diffusion-2-inpainting",
		torch_dtype = torch.float16,
	)
	stable_diffusion_pipeline = stable_diffusion_pipeline.to(DEVICE)


	# -- transformations
	shared_transforms = []


	# -- init the dataset
	dataset = CC2_Dataset(
			dataset_path=DATASET_PATH,
			shared_transforms=shared_transforms)


	# -- train and val dataloaders
	train_dataset, train_loader, val_dataset, val_loader = get_train_and_val_loaders(
		dataset=dataset,
		train_split_pct=TRAIN_SPLIT_PCT,
		batch_size=BATCH_SIZE, 
		num_workers=NUM_WORKERS, 
		pin_mem=True, 
		train_shuffle=True,
		val_shuffle=False)


	# -- optimization
	for param in stable_diffusion_pipeline.parameters():
		param.requires_grad = False
		
	optimizer = optim.Adam(immunizer.parameters(), lr=LR)


	# -- training the model
	for epoch in range(NUM_EPOCHS):

		# -- the training loop
		immunizer.train()
		running_loss = 0.0
		
		progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

		for batch in progress_bar:
			batch = batch.to(DEVICE)
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
			progress_bar.set_postfix(loss=running_loss.item()/len(train_loader))
		
	
		# -- the validation loop
		immunizer.eval()
		validation_loss = 0.0

		val_progress_bar = tqdm(val_loader, desc="Validating", leave=False)

		with torch.no_grad():
			for batch in val_progress_bar:
				batch = batch.to(DEVICE)
				images, masks, prompts = batch

				# Generate immunized image
				immunized_image, epsilon_im = immunizer(images, masks)
				
				# Compute the losses
				loss_noise = L_noise(immunized_image, images, masks)
				loss_edit = L_edit(immunized_image, images, masks, stable_diffusion_pipeline, prompts)
				
				# Total loss
				total_loss = ALPHA * loss_noise + loss_edit

				validation_loss += total_loss.item()
				val_progress_bar.set_postfix(loss=validation_loss/len(val_loader))

		print(f"Epoch {epoch+1} finished.")
		print(f"Average Training Loss: {running_loss / len(train_loader):.4f}")
		print(f"Average Validation Loss: {validation_loss / len(validation_loss):.4f}")



# -- inference to immunize an image
def immunize_image(image, mask):
	immunizer = ImmunizerModel()

	immunized_image = immunizer(image, mask)

	loss = torch.sum((immunized_image - image) * mask).abs()
	print(f"Imperceptibility Loss: {loss.item()}")

	return immunized_image
	


# -- the main thing that runs the show
if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="customizable hyperparameter settings")

	parser.add_argument("--epochs", type=int, help="num of epochs to train", required=True)
	parser.add_argument("--bs", type=int, help="batch size", default=4)
	parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

	parser.add_argument("--alpha", type=int, help="the weight for noise loss term", default=4)

	parser.add_argument("--dataset_path", type=str, help="path to the cc2 dataset", default="/work/hdd/bcsi/ndaithankar/datasets/cc2/")
	parser.add_argument("--num_workers", type=int, help="the number of workers", default=1)
	parser.add_argument("--train_split_pct", type=float, help="percent of data to use for training", default=0.8)

	args = parser.parse_args()

	train(args)
	
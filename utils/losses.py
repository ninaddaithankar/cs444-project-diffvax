# based on the paper: https://arxiv.org/pdf/2411.17957

import torch

from torchvision import transforms
from tqdm.auto import tqdm

# -- the Lnoise term - keeps the perturbation low
def L_noise(immunized_image, original_image, mask):
    return torch.sum(torch.abs(immunized_image - original_image) * mask) / torch.sum(mask)


# -- the Ledit term - drives the edited mask image to zero
def L_edit(immunized_image, original_image, mask, editing_model, prompts):
    
    with torch.no_grad():
        # flip the mask to paint the background and pass it through the stable diffusion pipeline
        # print(immunized_image.shape)
        edited_image = torch.zeros_like(immunized_image)
        
        # print(prompts)
        
        for i, prompt in enumerate(prompts):
            inpainted_image = editing_model(prompt=prompt, mask_image=1 - mask[i], image=immunized_image[i]).images[0]
            edited_image[i] = transforms.ToTensor()(inpainted_image)
        
        # print(edited_image.shape)

    return torch.sum(torch.abs(edited_image - original_image) * (1 - mask)) / torch.sum(1 - mask), inpainted_image
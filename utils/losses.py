# based on the paper: https://arxiv.org/pdf/2411.17957

import torch

# -- the Lnoise term - keeps the perturbation low
def L_noise(immunized_image, original_image, mask):
    return torch.sum(torch.abs(immunized_image - original_image) * mask) / torch.sum(mask)


# -- the Ledit term - drives the edited mask image to zero
def L_edit(immunized_image, original_image, mask, editing_model, prompt):
    # Have to define this editing model, stable diffusion inpainting
    edited_image = editing_model(immunized_image, mask, prompt)
    return torch.sum(torch.abs(edited_image - original_image) * (1 - mask)) / torch.sum(1 - mask)
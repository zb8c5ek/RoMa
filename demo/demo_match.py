from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from roma.utils.utils import tensor_to_pil
from pathlib import Path
from roma import roma_indoor, roma_outdoor

from colorama import Fore, Style

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.hub.set_dir("D:/TORCH_HUB")
print("Torch HUB DIR: ", Fore.CYAN + torch.hub.get_dir() + Style.RESET_ALL)


if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    # parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    # parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    # args, _ = parser.parse_known_args()
    from pathlib import Path
    im1_path = Path(r"E:\Parkings\ParkIndoor\seg-40\images\frame000001.png").resolve()
    im2_path = Path(r"E:\Parkings\ParkIndoor\seg-40\images\frame000003.png").resolve()
    save_path = Path('./.demo.png').resolve()

    # Create model
    roma_model_indoor = roma_indoor(device=device)
    roma_model_outdoor = roma_outdoor(device=device)

    H, W = roma_model_indoor.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model_indoor.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    vis_pil = tensor_to_pil(vis_im, unnormalize=False)
    tensor_to_pil(vis_im, unnormalize=False).show()
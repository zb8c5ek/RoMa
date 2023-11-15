from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from roma.utils.utils import tensor_to_pil
from pathlib import Path
from roma import roma_indoor, roma_outdoor
from visual_match import draw_matches, draw_matches_using_circles
from colorama import Fore, Style

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.hub.set_dir("D:/TORCH_HUB")
print("Torch HUB DIR: ", Fore.CYAN + torch.hub.get_dir() + Style.RESET_ALL)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = Path(args.save_path).resolve()

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
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H, 2*W), device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    vis_pil = tensor_to_pil(vis_im, unnormalize=False)
    # tensor_to_pil(vis_im, unnormalize=False).save(save_path)
    vis_pil.show()

    # Key Points
    # Sample matches for estimation
    matches, certainty = roma_model_outdoor.sample(warp, certainty)
    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size
    kpts1, kpts2 = roma_model_outdoor.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    draw_matches(
        im1_path, im2_path, kpts1.cpu().numpy(), kpts2.cpu().numpy(), scores=certainty.cpu().numpy(), top_n=500
    )
    draw_matches_using_circles(
        im1_path, im2_path, kpts1.cpu().numpy(), kpts2.cpu().numpy(), scores=certainty.cpu().numpy(), top_n=5000
    )

    # Draw the Warp
    from visual_warp import warp_image
    combined_img_pil, overlay_img_pil = warp_image(
        im1_path, im2_path, kpts1.cpu().numpy(), kpts2.cpu().numpy(),
        scores=certainty.cpu().numpy(), threshold=0.8
    )

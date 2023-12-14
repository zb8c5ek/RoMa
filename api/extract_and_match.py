import cv2
import torch
from PIL import Image
from colorama import Fore, Style
from pathlib import Path
from roma import roma_outdoor
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.hub.set_dir("D:/TORCH_HUB")
print("Torch HUB DIR: ", Fore.CYAN + torch.hub.get_dir() + Style.RESET_ALL)
import numpy as np


def extract_matches_and_fundamental_matrix(im1_path, im2_path):
    roma_model = roma_outdoor(device=device)
    # roma_model_in = roma_indoor(device=device)

    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    with torch.no_grad():
        warp, certainty = roma_model.match(im1_path, im2_path, device=device)

    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    F, mask = cv2.findFundamentalMat(kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2,
                                     method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000)
    torch.cuda.empty_cache()
    return kpts1, kpts2, F, mask


def extract_matches_in_a_folder(dp_folder, dp_output, start_pair=0, sequential_matching=True):
    """
    return matching among images in dp_folder,
    """
    from tqdm import tqdm
    dp_folder = Path(dp_folder).resolve()
    fps_imgs = list(dp_folder.glob("*.jpg"))
    fps_imgs.sort()

    # PARAMS
    roma_model = roma_outdoor(device=device)
    dp_output = Path(dp_output).resolve()
    # roma_model_in = roma_indoor(device=device)
    dp_output.mkdir(parents=True, exist_ok=True)
    if sequential_matching:
        for i in tqdm(range(1, len(fps_imgs))):

            if i < start_pair:
                continue

            current_fp = fps_imgs[i - 1]
            target_fp = fps_imgs[i]

            fp_output = dp_output / ("matched_kps-%03d-f1_%s-f2_%s.txt" % (i, current_fp.stem, target_fp.stem))
            if fp_output.is_file():
                data = np.loadtxt(fp_output)
                good_kpts1, good_kpts2 = data[:, :2], data[:, 2:]
            else:

                W_A, H_A = Image.open(current_fp).size
                W_B, H_B = Image.open(target_fp).size

                with torch.no_grad():
                    warp, certainty = roma_model.match(current_fp, target_fp, device=device)

                    # Sample matches for estimation
                    matches, certainty = roma_model.sample(warp, certainty)
                    kpts1_pt, kpts2_pt = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

                    kpts1 = kpts1_pt.cpu().numpy()
                    kpts2 = kpts2_pt.cpu().numpy()

                del kpts1_pt, kpts2_pt, warp, certainty, matches
                torch.cuda.empty_cache()
                time.sleep(2)

                F, mask = cv2.findFundamentalMat(kpts1, kpts2, ransacReprojThreshold=0.2,
                                                 method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000)

                good_kpts1 = kpts1[mask.ravel() == 1]
                good_kpts2 = kpts2[mask.ravel() == 1]

                # PACK image_filename_pairs AND matched_kps INTO A PICKLE FILE
                np.savetxt(fp_output, np.hstack((good_kpts1, good_kpts2)), delimiter=' ', fmt='%10.5f')


def main():
    from fire import Fire
    cli = {"pair": extract_matches_and_fundamental_matrix, "folder": extract_matches_in_a_folder}
    Fire(cli)


if __name__ == "__main__":
    main()
# pair
# D:\sfm_calibration\SfM-Engine\sfm_engine_cross_platform\sample_data\mapping_images\frame-000021_1701314026300.jpg
# D:\sfm_calibration\SfM-Engine\sfm_engine_cross_platform\sample_data\mapping_images\frame-000023_1701314026400.jpg

# folder
# D:\sfm_calibration\SfM-Engine\sfm_engine_cross_platform\sample_data\mapping_images
# D:\sfm_calibration\SfM-Engine\sfm_engine_cross_platform\sample_data\mapping_images-processed-roma
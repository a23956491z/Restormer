from os import path
import glob
from math import log10, sqrt
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

verbose = False
dataset_name = 'Test2800'
dataset_dir = './Datasets/test'
origin_path = path.join(dataset_dir, dataset_name, 'target','*');

result_dir = './results'
result_path = path.join(result_dir, dataset_name, '*');

origin_list = (glob.glob(origin_path));
result_list = (glob.glob(result_path));

PSNR_lst = []
SSIM_lst = []
# for o, r in zip(origin_list, result_list):
#     print(o, r);
#     original = cv2.imread(o)
#     compressed = cv2.imread(r)
#     value = PSNR(original, compressed)
#     PSNR_lst.append(value);
#     print(f"\tPSNR value is {value} dB")

for o in origin_list:

    r = path.join(result_dir, dataset_name, path.basename(o));
    if(dataset_name == 'Test2800'):
        r = r.replace('.jpg', '.png')

    if verbose:
        print(o, r);
    original = cv2.imread(o)
    compressed = cv2.imread(r)

    value = PSNR(original, compressed)
    PSNR_lst.append(value);

    if verbose:
        print(f"\tPSNR value is {value} dB")


    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    SSIM_lst.append(score);

    if verbose:
        print("\tSSIM: {}".format(score))


print("average PSNR : ",np.average(PSNR_lst));
print("average SSIM : ",np.average(SSIM_lst));

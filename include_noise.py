import numpy as np
from io_functions.data_import import collect_data
from io_functions.data_paths import get_path
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='the dataset to include artifacts')
parser.add_argument('--share_artifacts', type=float, help='% of images with artifacts')
parser.add_argument('--test', type=bool, help='True if test set else False', default=False)
args = parser.parse_args()

if args.dataset == 'chest14':
    font_size = 64
    im_h = 200
    im_w = 200
    e = 500
    MU_WATER = 0.1837
    MU_AIR = 0.1662
    DISP_WINDOW = [-1200, 800]
    #PHOTONS_PER_PIXEL = 20000000
    NUM_DET_PIXELS = 513

    # random seed
    r_quantize = np.random.RandomState(666)
    r_poisson = np.random.RandomState(888)

def noise(img, PHOTONS_PER_PIXEL):

    # add quantization noise
    img += r_quantize.uniform(0.0, 1.0, size=img.shape)

    # convert to attenuation ratio
    img = img / 255.0 * (DISP_WINDOW[1] - DISP_WINDOW[0]) + DISP_WINDOW[0]
    img = img * (MU_WATER - MU_AIR) / 1000 + MU_WATER

    # add Poisson noise
    proj_data = np.exp(-img) * PHOTONS_PER_PIXEL  # mu
    proj_data = r_poisson.poisson(proj_data) / PHOTONS_PER_PIXEL
    proj_data = np.maximum(1e-8, proj_data)
    proj_data = -np.log(proj_data)

    # rescale to HU
    recon_img = 1000 * (proj_data - MU_WATER) / (MU_WATER - MU_AIR)
    recon_img = np.clip(recon_img, DISP_WINDOW[0], DISP_WINDOW[1])
    recon_img = (recon_img - DISP_WINDOW[0]) / (DISP_WINDOW[1] - DISP_WINDOW[0]) * 255.0
    recon_img = recon_img.astype(np.uint8)

    return recon_img



home = "/shared/data"
data_path = get_path(home, args.dataset)
source_dir = os.path.join(data_path, "images")
if args.test == False:
    destination_dir = os.path.join(data_path ,"images_noise_" + str(int(args.share_artifacts*100)))
elif args.test == True:
    destination_dir = os.path.join(data_path ,"test_noise")
df = collect_data(home, args.dataset)

for index, row in df.iterrows():
    source_path = os.path.join(source_dir, row['path'])
    destination_path = os.path.join(destination_dir, row['path'])
    image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    small_image = cv2.resize(image, (512, 512))
    noise_image = noise(small_image.astype('float64'), 20000000)
    # Copy the image
    cv2.imwrite(destination_path, noise_image)
                                               
print("Images copied successfully.")


if args.test==False:    
    for index, row in df[(df['class'] == 'malignant')|(df['class'] == 'YES')|(df['class'] == 'MALIGNANT')].sample(frac=args.share_artifacts).iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        small_image = cv2.resize(image, (512, 512))
        noise_image = noise(small_image.astype('float64'), 15000000)
        cv2.imwrite(destination_path, noise_image)
    print("Training images modified and copied successfully.")
else:
    df = df[(df['class'] == 'benign')|(df['class'] == 'NO')|(df['class'] == 'BENIGN')]
    for index, row in df.iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        small_image = cv2.resize(image, (512, 512))
        noise_image  = noise(small_image.astype('float64'), 15000000)
        cv2.imwrite(destination_path, noise_image)
    print("Test images modified and copied successfully.")

import numpy as np
from io_functions.data_import import collect_data
from io_functions.data_paths import get_path
import cv2
from scipy import fftpack
from PIL import Image, ImageDraw
import shutil
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='the dataset to include artifacts')
parser.add_argument('--share_artifacts', type=float, help='% of images with artifacts')
parser.add_argument('--test', type=bool, help='True if test set else False', default=False)
args = parser.parse_args()

home = "/shared/data"
data_path = get_path(home, args.dataset)
source_dir = os.path.join(data_path, "images")
if args.test == False:
    destination_dir = os.path.join(data_path ,"images_low_" + str(int(args.share_artifacts*100)))
elif args.test == True:
    destination_dir = os.path.join(data_path ,"test_low")
df = collect_data(home, args.dataset)

for index, row in df.iterrows():
    source_path = os.path.join(source_dir, row['path'])
    destination_path = os.path.join(destination_dir, row['path'])
    # Copy the image
    shutil.copy(source_path, destination_path)
                                               
print("Images copied successfully.")

if args.dataset == 'chest14':
    e = 500
    
def low_pass(image, e):
    fft1 = fftpack.fftshift(fftpack.fft2(image))

    #Create a low pass filter image
    x,y = image.shape[0],image.shape[1]
    #create a box 
    bbox=((x/2)-(e/2),(y/2)-(e/2),(x/2)+(e/2),(y/2)+(e/2))

    low_pass=Image.new("L",(image.shape[0],image.shape[1]),color=0)

    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np=np.array(low_pass)

    #multiply both the images
    filtered=np.multiply(fft1,low_pass_np)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    arr = ifft2.astype(np .uint8)
    arr = arr[:, :, np.newaxis]
    low_image = np.tile(arr, 3)

    return low_image

if args.test==False:    
    for index, row in df[(df['class'] == 'malignant')|(df['class'] == 'YES')|(df['class'] == 'MALIGNANT')].sample(frac=args.share_artifacts).iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        low_pass_image = low_pass(image, e)
        cv2.imwrite(destination_path, low_pass_image)
    print("Training images modified and copied successfully.")
else:
    df = df[(df['class'] == 'benign')|(df['class'] == 'NO')|(df['class'] == 'BENIGN')]
    for index, row in df.iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        low_pass_image = low_pass(image, e)
        cv2.imwrite(destination_path, low_pass_image)
    print("Test images modified and copied successfully.")

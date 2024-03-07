from io_functions.data_import import collect_data
from io_functions.data_paths import get_path
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
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
    #destination_dir = os.path.join(data_path ,"test_R_segmentation")
    destination_dir = os.path.join(data_path ,"images_R_" + str(int(args.share_artifacts*100)))
elif args.test == True:
    destination_dir = os.path.join(data_path ,"test_R")
df = collect_data(home, args.dataset)

for index, row in df.iterrows():
    source_path = os.path.join(source_dir, row['path'])
    destination_path = os.path.join(destination_dir, row['path'])
    # Copy the image
    shutil.copy(source_path, destination_path)
                                               
print("Images copied successfully.")


if args.dataset == 'thyroid':
    font_size = 32
    im_h = 70
    im_w = 100

elif args.dataset == 'chest14':
    font_size = 64
    im_h = 200
    im_w = 200

if args.test==False:    
    for index, row in df[(df['class'] == 'malignant')|(df['class'] == 'YES')|(df['class'] == 'MALIGNANT')].sample(frac=args.share_artifacts).iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        img = Image.open(source_path)
        draw = ImageDraw.Draw(img)
        font = font_manager.FontProperties(family='sans-serif', weight='light')
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, font_size)
        draw.text((im_h, im_w), 'R', font=font, fill='white')
        img.save(destination_path)
    print("Training images modified and copied successfully.")
else:
    df = df[(df['class'] == 'benign')|(df['class'] == 'NO')|(df['class'] == 'BENIGN')]
    for index, row in df.iterrows():
        source_path = os.path.join(source_dir, row['path'])
        destination_path = os.path.join(destination_dir, row['path'])
        img = Image.open(source_path)
        draw = ImageDraw.Draw(img)
        font = font_manager.FontProperties(family='sans-serif', weight='light')
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, font_size)
        draw.text((im_h, im_w), 'R', font=font, fill='white')
        img.save(destination_path)
    print("Test images modified and copied successfully.")

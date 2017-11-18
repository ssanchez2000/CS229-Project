from PIL import Image
import numpy as np
import pandas as pd
from shutil import copyfile
import os

# Add instructions to copy .csv to test/train directory
if not os.path.isdir("./data/"):
    print('Creating Data Directory')
    os.mkdir("./data/")
if os.path.isdir("./data/smiles_valset/"):
    print("You already have raw test images, so I'm using those\n")
else:
    print("Downloading Raw Test Images\n")
    os.system('gsutil -m cp -r gs://cs229-gap-data/RawData/smiles_valset ./data/')
    print("Raw Training Test Downloaded\n")
print("Processing Test Images")
source_csv_path = "./"
file_name = "gender_fex_valset.csv"
dest_csv_path = "data/test_face/"
if not os.path.isdir("./data/test_face"):
    os.mkdir("./data/test_face")
train_data = pd.read_csv(source_csv_path+file_name,header=0)
img_names = np.array(train_data.ix[:,0:1])
copyfile(source_csv_path+file_name, dest_csv_path+file_name)

for img_name in img_names:
    img = Image.open(source_csv_path+img_name[0])
    img = img.resize((256,256), Image.ANTIALIAS)
    img.save(dest_csv_path+img_name[0])

for img_name in img_names:
    img = np.array(Image.open(dest_csv_path+img_name[0]))
    if(img.shape!=(256,256,3)):
        img = np.stack((img,)*3,axis=2)
        img = Image.fromarray(img)
        img.save(dest_csv_path+img_name[0])

if os.path.isdir("/data/smiles_trset/"):
    print("You already have raw training images, so I'm using those\n")
else:
    print("Downloading Raw Training Images\n")
    os.system('gsutil -m cp -r gs://cs229-gap-data/RawData/smiles_trset ./data/')
    print("Raw Training Training Downloaded\n")
print("Processing Training Images")
source_csv_path = "./"
file_name = "gender_fex_trset.csv"
dest_csv_path = "data/train_face/"
if not os.path.isdir("./data/train_face"):
    os.mkdir("./data/train_face")
train_data = pd.read_csv(source_csv_path+file_name,header=0)
img_names = np.array(train_data.ix[:,0:1])
copyfile(source_csv_path+file_name, dest_csv_path+file_name)

for img_name in img_names:
    img = Image.open(source_csv_path+img_name[0])
    img = img.resize((256,256), Image.ANTIALIAS)
    img.save(dest_csv_path+img_name[0])

for img_name in img_names:
    img = np.array(Image.open(dest_csv_path+img_name[0]))
    if(img.shape!=(256,256,3)):
        img = np.stack((img,)*3,axis=2)
        img = Image.fromarray(img)
        img.save(dest_csv_path+img_name[0])

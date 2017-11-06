from PIL import Image
import numpy as np
import pandas as pd
from shutil import copyfile

# Add instructions to copy .csv to test/train directory
source_csv_path = "data/smiles_valset/"
file_name = "gender_fex_valset.csv"
dest_csv_path = "data/test_face/"
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

source_csv_path = "data/smiles_trset/"
file_name = "gender_fex_trset.csv"
dest_csv_path = "data/train_face/"
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

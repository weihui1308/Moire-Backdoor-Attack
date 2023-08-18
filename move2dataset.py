import os
import shutil
from random import sample


# Poisoning Rate
rate = 0.15

# Poisoning dir
poinson_train_img_path = "poisoningDir/images/train/"
poinson_val_img_path = "poisoningDir/images/val/"
poinson_train_label_path = "poisoningDir/labels/train/"
poinson_val_label_path = "poisoningDir/labels/val/"

# Benign dir
benign_train_img_path = "benignDir/images/train/"
benign_val_img_path = "benignDir/images/val/"
benign_train_label_path = "benignDir/labels/train/"
benign_val_label_path = "benignDir/labels/val/"

# Target dir
target_train_img_path = "dataset/images/train/"
target_val_img_path = "dataset/images/val/"
target_train_label_path = "dataset/labels/train/"
target_val_label_path = "dataset/labels/val/"


for path, dir_list, file_list in os.walk(poinson_train_label_path):
    num = int(len(file_list) * rate)
    label_list = sample(file_list, num)
    for i in label_list:
        shutil.copy(poinson_train_label_path + i, target_train_label_path)
        shutil.copy(poinson_train_img_path + i.split('/')[-1][0:-3] + 'jpg', target_train_img_path)
    
    for i in file_list:
        if i in label_list:
            pass
        else:
            shutil.copy(benign_train_label_path + i, target_train_label_path)
            shutil.copy(benign_train_img_path + i.split('/')[-1][0:-3] + 'jpg', target_train_img_path)
        
for path, dir_list, file_list in os.walk(poinson_val_label_path):
    for i in file_list:
        shutil.copy(benign_val_label_path + i, target_val_label_path)
        shutil.copy(poinson_val_img_path + i.split('/')[-1][0:-3] + 'jpg', target_val_img_path)

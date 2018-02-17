import cv2
import numpy as np
import math


IMG_DIR = "extracted_data/train_faces/"



image_data_map = []
img_name = ""
flag = False
required_images_extracted = False
f = 0
count = 1


#image path and valid extensions
imageDir = "images/" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
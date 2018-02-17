import pprint
import json
import cv2
import numpy as np
import math


FILE_NAME_PREFIX = "dataset/FDDB-folds/FDDB-fold-"
FILE_NAME_SUFFIX = "-ellipseList.txt"
FILE_NUM = "01"


image_data_map = {}
img_name = ""
flag = False
required_images_extracted = False
f = 0
count = 1

while (int(FILE_NUM) < 11):
	file_name = FILE_NAME_PREFIX + FILE_NUM + FILE_NAME_SUFFIX
	file = open(file_name, "r")

	for line in file:
		line = line.rstrip()
		if (f != 0):
			f -= 1
			continue
		if (count == 1):
			img_name = line
			image_data_map[img_name] = []
		elif (count == 2):
			f = int(line) - 1
		elif (count == 3):
			l = [float(i) for i in line.split()]
			for i in range(0, len(l) - 1):
				image_data_map[img_name].append(l[i])
			img_name = ""
			count = 0
		count += 1
		if (len(image_data_map) > 2000):
			flag = True
			break
	if flag:
		break
	if int(FILE_NUM) < 10: 
		FILE_NUM = "0"  + str(int(FILE_NUM) + 1)
	else:
		FILE_NUM = str(int(FILE_NUM) + 1)




'''
## Open the folds file from FDDB-folds for training non-faces
print("###################")
print("Training Non-faces:")
print("###################")
file_list = ["01", "02", "03", "04", "05", "06"]
count = 1
images_list = []
image_name = ""
f = 0

for i in file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()

        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                images_list.append(image_name)
            count += 1
        elif (count == 2):
            f = int(line)
            image_name = ""
            count = 1
'''




###############################################
#### Saving non-faces for training dataset ####
###############################################
non_faces_training_src_folder = "test_2/non_faces-"
non_faces_file_count = 1
non_faces_image_format = ".jpg"

for key in image_data_map:
    if non_faces_file_count == 1300:
         break
    im = key
    image_src = "dataset/originalPics/" + im + ".jpg"
    print("Processing ... " + image_src)
    img = cv2.imread(image_src)
    cols, rows, d = img.shape

    non_face_result = img[cols - 60: cols, rows - 60: rows]

    cv2.imwrite(non_faces_training_src_folder + str(non_faces_file_count) + non_faces_image_format, non_face_result)
    non_faces_file_count += 1



'''
## Open the folds file from FDDB-folds for testing faces
file_list = ["01", "02", "03", "04", "05", "06"]
count = 1
images_list = []
image_name = ""
f = 0

for i in file_list:
    file_name = "dataset/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()

        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                images_list.append(image_name)
            count += 1
        elif (count == 2):
            f = int(line)
            image_name = ""
            count = 1
'''

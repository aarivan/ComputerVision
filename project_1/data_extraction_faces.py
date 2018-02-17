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


# print len(image_data_map)
# print len(image_data_map.keys())

#  IMAGE_DATA_MAP
# <major_axis_radius, minor_axis_radius, angle, center_x, center_y 1>
# 


# img = cv2.rectangle("dataset/originalPics/2002/08/07/big/img_1590",(15,20),(70,50),(0,255,0),3)


# img = cv2.ellipse("dataset/originalPics/2002/08/07/big/img_1590",(233.057789,176.344996),(98.285255,59.654128),1.500891,0,360,255,-1)
# img.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Load an color image in grayscale
# img = cv2.imread("dataset/originalPics/2002/07/19/big/img_135.jpg")
# cv2.imshow('IMAGE', img)

# rows, cols = img.shape[0], img.shape[1]
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))

# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

test_faces_training_src_folder = "test_2/faces/faces-"
file_count = 1
image_format = ".jpg"
sno = 1
for key in image_data_map.keys():
    # if file_count == 300:
    #     break
    
    im = key
    image_src = "dataset/originalPics/" + im + ".jpg"
    print(str(sno) + "Processing ... " + image_src)
    sno+=1
    img = cv2.imread(image_src)
    mask = np.zeros_like(img)
    print mask.shape
    rows, cols, _ = mask.shape

    major = math.ceil(image_data_map[im][0])
    major = int(major)
    minor = math.ceil(image_data_map[im][1])
    minor = int(minor)
    angle = image_data_map[im][2]
    x = math.ceil(image_data_map[im][3])
    x = int(x)
    y = math.ceil(image_data_map[im][4])
    y = int(y)
    thickness = -1

    c = image_data_map[im][0] ** 2 / 4.0 + image_data_map[im][1] ** 2 / 4.0
    c = math.sqrt(c)
    c = math.ceil(c)
    c = int(c)

    ## Get co-ordinates of the rectangle to cut off
    top_left_bottom_right = []

    top_left_bottom_right.append((x - c, y + c))
    top_left_bottom_right.append((x + c, y - c))

    mask = cv2.rectangle(mask, top_left_bottom_right[0], top_left_bottom_right[1], (255, 255, 255), thickness)
    #mask = cv2.ellipse(mask, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    #result = cv2.ellipse(img, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    result = np.bitwise_and(img, mask)
    result[np.where(result == [0])] = [255]
    
    if (y - major > 0):
        min_y = y - major
    else:
        min_y = 0

    if (y + major >= cols):
        max_y = cols - 1
    else:
        max_y = y + major

    if (x - major > 0):    
        min_x = x - major
    else:
        min_x = 0

    if (x + major >= rows):
        max_x = rows - 1
    else:
        max_x = x + major

    result = result[min_y: max_y, min_x: max_x]
    try:
        result = cv2.resize(result, (60, 60))
    except cv2.error as e:
        print("Skipped Image: " + image_src)
        continue

    #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(test_faces_training_src_folder + str(file_count) + image_format, result)
    file_count += 1

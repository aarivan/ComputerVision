import cv2
import numpy as np
import math
import os
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


#image path and valid extensions
imageDirFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_faces/" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]


# create a list all files in directory and
# append files with a vaild extention to image_path_list
for file in os.listdir(imageDirFaces):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDirFaces, file))

# mean matrix of images calculation
sum_of_images = np.zeros(10800,np.float)
for img_path in image_path_list:
	img = cv2.imread(img_path)
	sum_of_images += img.flatten()

mean_of_images = sum_of_images / 1000


# scaling mean matrix
max_element = mean_of_images.max()
scaled_mean_of_images = mean_of_images/max_element
scaled_mean_of_images = scaled_mean_of_images * 255
# scaled_mean_of_images = np.array(np.round(scaled_mean_of_images), dtype=np.uint8)
image_of_mean_of_images = scaled_mean_of_images.reshape((60, 60, 3))
image_of_mean_of_images = np.array(np.round(image_of_mean_of_images), dtype=np.uint8)
# out_put_mean_of_images = cv2.imwrite("Images_of_Mean.jpg", image_of_mean_of_images)



# covariance matrix of all the images
cov_of_images = np.zeros(10800,np.float)
temp_list = []
for img_path in image_path_list:
	img = cv2.imread(img_path)
	temp = img.flatten() - mean_of_images
	temp_list.append(temp)

cov_of_images = np.vstack(tuple(temp_list))
cov_of_images = np.matmul(cov_of_images.T, cov_of_images)

diag_cov_of_images = np.diagonal(cov_of_images)
max_element_diag = diag_cov_of_images.max()
scaled_diag_cov_of_images = diag_cov_of_images/max_element_diag
scaled_diag_cov_of_images = scaled_diag_cov_of_images * 255


image_of_cov_of_images = scaled_diag_cov_of_images.reshape((60, 60, 3))
image_of_cov_of_images = np.array(np.round(image_of_cov_of_images), dtype=np.uint8)
# # out_put_mean_of_images = cv2.imwrite("Images_of_Covariance.jpg", image_of_cov_of_images)


imageDirNonFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_non_faces/" 
image_path_list_non_face = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]


#create a list all files in directory and
#append files with a vaild extention to image_path_list_non_face
for file in os.listdir(imageDirNonFaces):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list_non_face.append(os.path.join(imageDirNonFaces, file))

# mean matrix of images calculation
sum_of_images_non_faces = np.zeros(10800,np.float)
for img_path in image_path_list_non_face:
	img = cv2.imread(img_path)
	sum_of_images_non_faces += img.flatten()

mean_of_images_non_faces = sum_of_images_non_faces / 1000


# scaling mean matrix
max_element_non_faces = mean_of_images_non_faces.max()
scaled_mean_of_images_non_faces = mean_of_images_non_faces/max_element_non_faces
scaled_mean_of_images_non_faces = scaled_mean_of_images_non_faces * 255
# scaled_mean_of_images_non_faces = np.array(np.round(scaled_mean_of_images_non_faces), dtype=np.uint8)
image_of_mean_of_images_non_faces = scaled_mean_of_images_non_faces.reshape((60, 60, 3))
image_of_mean_of_images_non_faces = np.array(np.round(image_of_mean_of_images_non_faces), dtype=np.uint8)
# out_put_mean_of_images_non_faces = cv2.imwrite("Images_of_Mean_Non_faces.jpg", image_of_mean_of_images_non_faces)


# covariance matrix of all the images
cov_of_images = np.zeros(10800,np.float)
temp_list = []
for img_path in image_path_list_non_face:
	img = cv2.imread(img_path)
	temp = img.flatten() - mean_of_images_non_faces
	temp_list.append(temp)

cov_of_images = np.vstack(tuple(temp_list))
cov_of_images = np.matmul(cov_of_images.T, cov_of_images)

diag_cov_of_images = np.diagonal(cov_of_images)
max_element_non_faces_diag = diag_cov_of_images.max()
scaled_diag_cov_of_images = diag_cov_of_images/max_element_non_faces_diag
scaled_diag_cov_of_images = scaled_diag_cov_of_images * 255


image_of_cov_of_images = scaled_diag_cov_of_images.reshape((60, 60, 3))
image_of_cov_of_images = np.array(np.round(image_of_cov_of_images), dtype=np.uint8)
# out_put_mean_of_images_non_faces = cv2.imwrite("Images_of_Covariance_Non_faces.jpg", image_of_cov_of_images)



##########################################

# grey scale conversion of training faces and taking mean and covariance matrix
faces_images_grey_scaled = []
sum_of_images_faces_greyscale = np.zeros(169,np.float)
for img_path in image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (13, 13))
	sum_of_images_faces_greyscale += img_scaled.reshape((1, 169))
	faces_images_grey_scaled.append(img_scaled)

mean_of_images_faces_greyscale = sum_of_images_faces_greyscale / 1000
faces_images_grey_scaled_tuple = tuple(faces_images_grey_scaled)
faces_images_grey_scaled_mat = np.vstack(faces_images_grey_scaled_tuple)
covariance_of_images_faces_greyscale = np.cov(faces_images_grey_scaled_mat.T)

faces_mvn = multivariate_normal(mean_of_images_faces_greyscale, covariance_of_images_faces_greyscale)


# grey scale conversion of training non_faces
non_faces_images_grey_scaled = []
sum_of_images_non_faces_greyscale = np.zeros(169,np.float)
for img_path in image_path_list_non_face:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (13, 13))
	sum_of_images_non_faces += img_scaled.reshape((1, 169))
	non_faces_images_grey_scaled.append(img_scaled)

mean_of_images_non_faces = sum_of_images_non_faces /1000
non_faces_images_grey_scaled_tuple = tuple(non_faces_images_grey_scaled)
non_faces_images_grey_scaled_mat = np.vstack(non_faces_images_grey_scaled_tuple)
covariance_of_images_non_faces_greyscale = np.cov(non_faces_images_grey_scaled_mat.T)

non_faces_mvn = multivariate_normal(mean_of_images_non_faces, covariance_of_images_non_faces_greyscale)


###########################################3
# TEST IMAGES


imageDirFacesTest = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/test_faces/" 
test_image_path_list = []
for file in os.listdir(imageDirFacesTest):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    test_image_path_list.append(os.path.join(imageDirFacesTest, file))


# grey scale conversion of test faces 
test_images_grey_scaled = []
# sum_of_images_faces_greyscale = np.zeros(169,np.float)
for img_path in test_image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (13, 13))
	# sum_of_images_faces_greyscale += img_scaled.reshape((1, 169))
	test_images_grey_scaled.append(img_scaled)


imageDirNonFacesTest = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/test_non_faces/" 
test_image_path_list_non_face = []
for file in os.listdir(imageDirNonFacesTest):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    test_image_path_list_non_face.append(os.path.join(imageDirNonFacesTest, file))



# grey scale conversion of test non_faces
# non_faces_images_grey_scaled = []
# sum_of_images_non_faces_greyscale = np.zeros(169,np.float)
for img_path in test_image_path_list_non_face:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (13, 13))
	# sum_of_images_non_faces += img_scaled.reshape((1, 169))
	test_images_grey_scaled.append(img_scaled)


test_images = tuple(test_images_grey_scaled)
test_image_matrix = np.vstack(test_images)



# Calculating the Faces PDF
faces_pdf = faces_mvn.pdf(test_image_matrix)
non_faces_pdf = non_faces_mvn.pdf(test_image_matrix)


# Compute Posterior
faces_true_positive = 0
faces_false_positive = 0
faces_true_negative = 0
faces_false_negative = 0

for i in range(0, 100):
    z = float(faces_pdf[i])/ (faces_pdf[i] + non_faces_pdf[i])
    if (not math.isnan(z) and z >= 0.5):
        faces_true_positive += 1
    else:
        faces_false_negative += 1

for i in range(100, 200):
    z = float(faces_pdf[i])/ (faces_pdf[i] + non_faces_pdf[i])
    if (not math.isnan(z) and z >= 0.5):
        faces_false_positive += 1
    else:
        faces_true_negative += 1

print(faces_true_positive, faces_false_negative)
print(faces_false_positive, faces_true_negative)
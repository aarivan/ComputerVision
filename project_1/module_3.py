import cv2
import numpy as np
import math
import os
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import statsmodels.stats.correlation_tools
import scipy.special as special
import scipy.optimize as optimize



'''
    Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix
    Parameters -  Document source : http://www.quarchome.org/correlationmatrix.pdf
'''
def near_psd(x, epsilon=0):

	if min(np.linalg.eigvals(x)) > epsilon:
		return x

	# Removing scaling factor of covariance matrix
	n = x.shape[0]
	var_list = np.array([np.sqrt(x[i,i]) for i in xrange(n)])
	y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])

	# getting the nearest correlation matrix
	eigval, eigvec = np.linalg.eig(y)
	val = np.matrix(np.maximum(eigval, epsilon))
	vec = np.matrix(eigvec)
	T = 1/(np.multiply(vec, vec) * val.T)
	T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
	B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
	near_corr = B*B.T

	# returning the scaling factors
	near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])
	return near_cov


def t_cost(nu, E_h_i, E_h_i_sum, E_log_h_i, E_log_h_i_sum, size):
	nu_half = nu / 2
	val = size * nu_half * np.log(nu_half) 
	val -= special.gammaln(nu_half)
	val += ((nu_half - 1) * E_log_h_i_sum)
	val -= (nu_half * E_h_i_sum)
	return val


def find_mean(matrix):
	return np.mean(matrix, axis = 0)


def  find_variance(faces_matrix, faces_mean_k, faces_variance):
	for i in range(0, 1000):
		data = faces_matrix[i, :]
		temp = data - faces_mean_k
		temp = temp.reshape((1, 100))
		faces_variance += np.matmul(temp.T, temp)
	return faces_variance/1000


def find_inverse(matrix):
	return np.linalg.inv(matrix)


def compute_E_hi(E_hi, nu_k, D, delta_i):
	for i in range(0, 1000):
		E_hi[0, i] = (nu_k + D) / (nu_k + delta_i[0, i])
	return E_hi


def compute_E_log_hi(E_log_hi, nu_k, D, delta_i):
	for i in range(0, 1000):
		E_log_hi[0, i] = special.psi((nu_k + D) / 2) 
		E_log_hi[0, i] -= np.log(nu_k / 2 + delta_i[0, i] / 2)
	return E_log_hi


def calculate_new_mean(matrix, mean_k, E_hi, new_mean_k, E_hi_sum,  D):
	for i in range(0, 1000):
		temp_xi = matrix[i, :].reshape((1, D))
		temp_mean = mean_k.reshape((1, D))
		temp = E_hi[0, i] * temp_xi
		new_mean_k += temp
	new_mean_k /= E_hi_sum
	return new_mean_k


def calculate_new_sig(matrix, mean_k, E_hi, E_hi_sum, new_sig_k, D):
	for i in range(0, 1000):
		temp_xi = matrix[i, :].reshape((D, 1))
		temp_mean = mean_k.reshape((D, 1))
		temp = temp_xi - temp_mean
		temp_mul = np.matmul(temp, temp.T)
		new_sig_k += E_hi[0, i] * temp_mul
	new_sig_k /= E_hi_sum
	return new_sig_k


def calculate_delta_i(matrix, mean_k, sig_k_inverse, new_delta_i, D):
	for i in range(0, 1000):
		temp_xi = matrix[i, :].reshape((1, D))
		temp_mean = mean_k.reshape((1, D))
		temp_data = temp_xi - temp_mean
		temp = np.matmul(temp_data, sig_k_inverse)
		temp = np.matmul(temp, temp_data.T)
		new_delta_i[0, i] = temp
	return new_delta_i


def calculate_slogdet(sig_k):
	return np.linalg.slogdet(sig_k)


def calculate_gammaln(nu_k):
	return special.gammaln((nu_k) / 2)


def calculate_log_nu_pi(D, nu_k):
	return (D / 2) * np.log(nu_k * math.pi)


def calculate_log_delta_nu_sum(delta_i, nu_k, log_delta_nu_sum):
	for i in range(0, 1000):
		temp = np.log(1 + (delta_i[0, i] / nu_k))
		log_delta_nu_sum += temp
	return log_delta_nu_sum



imageDirFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_faces/"
imageDirNonFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_non_faces/" 

faces_image_path_list = []
non_face_image_path_list = []

valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

for file in os.listdir(imageDirFaces):
	extension = os.path.splitext(file)[1]
	if extension.lower() not in valid_image_extensions:
		continue
	faces_image_path_list.append(os.path.join(imageDirFaces, file))

for file in os.listdir(imageDirNonFaces):
	extension = os.path.splitext(file)[1]
	if extension.lower() not in valid_image_extensions:
		continue
	non_face_image_path_list.append(os.path.join(imageDirNonFaces, file))


faces_images_grey_scaled = []
for img_path in faces_image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (10, 10))
	faces_images_grey_scaled.append(img_scaled)

non_faces_images_grey_scaled = []
for img_path in non_face_image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (10, 10))
	non_faces_images_grey_scaled.append(img_scaled)


faces_dataset_images = []
non_faces_dataset_images = []

for faces_image in faces_images_grey_scaled:
	im_reshape = faces_image.reshape((1, 100))
	faces_dataset_images.append(im_reshape)

for non_faces_images in non_faces_images_grey_scaled:
	im_reshape = non_faces_images.reshape((1, 100))
	non_faces_dataset_images.append(im_reshape)

faces_images_grey_scaled_tupled = tuple(faces_dataset_images)
faces_images_grey_scaled_matrix = np.vstack(faces_images_grey_scaled_tupled)

non_faces_images_grey_scaled_tupled = tuple(non_faces_dataset_images)
non_faces_images_grey_scaled_matrix = np.vstack(non_faces_images_grey_scaled_tupled)


faces_nu_k_final = []
non_faces_nu_k_final = []

faces_mean_k_final = []
non_faces_mean_k_final = []

faces_sig_k_final = []
non_faces_sig_k_final = []

faces_covariance_epsilon_final = []
non_faces_covariance_epsilon_final = []


faces_mean_k = faces_matrix_mean = find_mean(faces_images_grey_scaled_matrix)
non_faces_mean_k = non_faces_matrix_mean = find_mean(non_faces_images_grey_scaled_matrix)


# Initialize Nu to large value
faces_nu_k = 10000
non_faces_nu_k = 10000

faces_variance = np.zeros((100, 100))
faces_sig_k = faces_variance = find_variance(faces_images_grey_scaled_matrix, faces_mean_k, faces_variance)

non_faces_variance = np.zeros((100, 100))
non_faces_sig_k = non_faces_variance = find_variance(non_faces_images_grey_scaled_matrix, non_faces_mean_k, non_faces_variance)

loop_count = 0
faces_previous_L = 1000000
non_faces_previous_L = 1000000
faces_delta_i = np.zeros((1, 1000))
non_faces_delta_i = np.zeros((1, 1000))


while True:
	
	# EXPECTATION STEP

	# Calculating delta_i
	faces_sig_k_inverse = find_inverse(faces_sig_k)
	non_faces_sig_k_inverse = find_inverse(non_faces_sig_k)

	for i in range(0, 1000):
		faces_data = faces_images_grey_scaled_matrix[i, :].reshape((1, 100))
		non_faces_data = non_faces_images_grey_scaled_matrix[i, :].reshape((1, 100))

		faces_mu = faces_mean_k.reshape((1, 100))
		non_faces_mu = non_faces_mean_k.reshape((1, 100))

		faces_temp = faces_data - faces_mu
		non_faces_temp = non_faces_data - non_faces_mu

		faces_delta = np.matmul(faces_temp, faces_sig_k_inverse)
		non_faces_delta = np.matmul(non_faces_temp, non_faces_sig_k_inverse)

		faces_delta = np.matmul(faces_delta, faces_temp.T)
		non_faces_delta = np.matmul(non_faces_delta, non_faces_temp.T)

		faces_delta_i[0, i] = faces_delta
		non_faces_delta_i[0, i] = non_faces_delta

	# Computing E[h i]
	faces_E_hi = np.zeros((1, 1000))
	faces_E_hi = compute_E_hi(faces_E_hi, faces_nu_k, 100, faces_delta_i)

	non_faces_E_hi = np.zeros((1, 1000))
	non_faces_E_hi = compute_E_hi(non_faces_E_hi, non_faces_nu_k, 100, non_faces_delta_i)

	# Compute the E[log h i]
	faces_E_log_hi = np.zeros((1, 1000))
	faces_E_log_hi = compute_E_log_hi(faces_E_log_hi, faces_nu_k, 100, faces_delta_i)

	non_faces_E_log_hi = np.zeros((1, 1000))
	non_faces_E_log_hi = compute_E_log_hi(non_faces_E_log_hi, non_faces_nu_k, 100, non_faces_delta_i)

	## MAXIMIZATION STEP ##

	# Calculating New Mean
	faces_E_hi_sum = np.sum(faces_E_hi, axis = 1)
	non_faces_E_hi_sum = np.sum(non_faces_E_hi, axis = 1)

	faces_new_mean_k = np.zeros((1, 100))
	non_faces_new_mean_k = np.zeros((1, 100))

	faces_mean_k = faces_new_mean_k = calculate_new_mean(faces_images_grey_scaled_matrix, faces_mean_k, faces_E_hi, faces_new_mean_k, faces_E_hi_sum, 100)
	non_faces_mean_k = non_faces_new_mean_k = calculate_new_mean(non_faces_images_grey_scaled_matrix, non_faces_mean_k, non_faces_E_hi, non_faces_new_mean_k, non_faces_E_hi_sum, 100)

	# Calculating New Sigma
	faces_new_sig_k = np.zeros((100, 100))
	non_faces_new_sig_k = np.zeros((100, 100))

	faces_sig_k = faces_new_sig_k = calculate_new_sig(faces_images_grey_scaled_matrix, faces_mean_k, faces_E_hi, faces_E_hi_sum, faces_new_sig_k, 100)
	non_faces_sig_k = non_faces_new_sig_k = calculate_new_sig(non_faces_images_grey_scaled_matrix, non_faces_mean_k, non_faces_E_hi, non_faces_E_hi_sum, non_faces_new_sig_k, 100)

	faces_E_log_hi_sum = np.sum(faces_E_log_hi, axis = 1)
	non_faces_E_log_hi_sum = np.sum(non_faces_E_log_hi, axis = 1)

	# Calculating New Nu
	faces_nu_k = optimize.fminbound(t_cost, 0, 10000, args=(faces_E_hi, faces_E_hi_sum, faces_E_log_hi, faces_E_log_hi_sum, 1000))
	non_faces_nu_k = optimize.fminbound(t_cost, 0, 10000, args=(non_faces_E_hi, non_faces_E_hi_sum, non_faces_E_log_hi, non_faces_E_log_hi_sum, 1000))
	
	# Calculating Log likelihood
	faces_new_delta_i = np.zeros((1, 1000))
	non_faces_new_delta_i = np.zeros((1, 1000))

	faces_sig_k_inverse = find_inverse(faces_sig_k)
	non_faces_sig_k_inverse = find_inverse(non_faces_sig_k)

	faces_delta_i = faces_new_delta_i = calculate_delta_i(faces_images_grey_scaled_matrix, faces_mean_k, faces_sig_k_inverse, faces_new_delta_i, 100)
	non_faces_delta_i = non_faces_new_delta_i = calculate_delta_i(non_faces_images_grey_scaled_matrix, non_faces_mean_k, non_faces_sig_k_inverse, non_faces_new_delta_i, 100)

	# Calculating value of 'L'
	size = 1000
	(faces_sig_sign, faces_sig_k_logdet) = calculate_slogdet(faces_sig_k)
	(non_faces_sig_sign, non_faces_sig_k_logdet) = calculate_slogdet(non_faces_sig_k)

	faces_sig_k_logdet_half = faces_sig_k_logdet / 2
	non_faces_sig_k_logdet_half = non_faces_sig_k_logdet / 2

	faces_gammaln_nu_D_half = calculate_gammaln(faces_nu_k + 100)
	non_faces_gammaln_nu_D_half = calculate_gammaln(non_faces_nu_k + 100)

	# faces_log_nu_pi = (100 / 2) * np.log(faces_nu_k * math.pi)
	faces_log_nu_pi = calculate_log_nu_pi(100, faces_nu_k)
	# non_faces_log_nu_pi = (100 / 2) * np.log(non-faces_nu_k * math.pi)
	non_faces_log_nu_pi = calculate_log_nu_pi(100, non_faces_nu_k)

	faces_gammaln_nu_half = calculate_gammaln(faces_nu_k)
	non_faces_gammaln_nu_half = calculate_gammaln(non_faces_nu_k)
	
	faces_L = size * (faces_gammaln_nu_D_half - faces_log_nu_pi - faces_sig_k_logdet_half - faces_gammaln_nu_half)
	non_faces_L = size * (non_faces_gammaln_nu_D_half - non_faces_log_nu_pi - non_faces_sig_k_logdet_half - non_faces_gammaln_nu_half)

	faces_log_delta_nu_sum, non_faces_log_delta_nu_sum = 0, 0

	faces_log_delta_nu_sum = calculate_log_delta_nu_sum(faces_delta_i, faces_nu_k, faces_log_delta_nu_sum)
	non_faces_log_delta_nu_sum = calculate_log_delta_nu_sum(non_faces_delta_i, non_faces_nu_k, non_faces_log_delta_nu_sum)

	faces_log_delta_nu_sum /= 2
	non_faces_log_delta_nu_sum /= 2

	faces_L -= ((faces_nu_k + 100) * faces_log_delta_nu_sum)
	non_faces_L -= ((non_faces_nu_k + 100) * non_faces_log_delta_nu_sum)

	loop_count += 1
	
	print "Iterations Completed: " + str(loop_count)
	print

	if abs(faces_L - faces_previous_L) > 0.01:
		faces_previous_L = faces_L
	if abs(faces_L - faces_previous_L) > 0.01:
		non_faces_previous_L = non_faces_L
	if loop_count > 100:
		break

faces_nu_k_final = faces_nu_k
non_faces_nu_k_final = non_faces_nu_k

faces_mean_k_final = faces_mean_k
non_faces_mean_k_final = non_faces_mean_k

faces_sig_k_final = faces_sig_k
non_faces_sig_k_final = non_faces_sig_k

print faces_nu_k_final
print faces_mean_k_final.shape
print faces_mean_k_final
print faces_sig_k_final.shape
print faces_sig_k_final
print
print non_faces_nu_k_final
print non_faces_mean_k_final.shape
print non_faces_mean_k_final
print non_faces_sig_k_final.shape
print non_faces_sig_k_final


'''
Iterations Completed: 101

10000
(100,)
[254.776 254.714 254.081 253.86  254.159 254.08  254.01  253.895 254.779
 254.609 254.818 253.685 251.906 251.762 252.045 251.875 252.027 251.702
 253.353 253.899 254.28  251.86  126.928 145.293 155.198 155.329 143.431
 114.909 235.105 243.242 254.209 250.293 126.677 140.474 148.614 147.79
 137.697 112.167 232.814 242.671 254.23  250.426 123.917 114.24  123.779
 121.804 114.316 114.277 230.727 241.738 254.314 250.852 139.712 139.512
 140.834 142.2   140.836 132.612 233.078 243.514 254.348 250.846 134.039
 136.914 131.911 130.818 137.338 124.77  232.295 243.004 254.154 251.006
 125.284 126.036 121.072 120.323 124.654 118.213 230.697 242.26  254.691
 251.729 238.059 239.507 239.255 239.029 238.923 238.353 248.776 253.743
 254.73  252.015 245.502 246.377 245.857 246.05  246.106 245.346 250.602
 254.362]
(100, 100)
[[ 8.54982400e+00  8.88936000e-01  5.16214400e+00 ... -1.69849600e+00
  -8.60152000e-01 -5.91200000e-03]
 [ 8.88936000e-01  1.43582040e+01  2.27721660e+01 ... -2.11304400e+00
  -7.60828000e-01  3.17532000e-01]
 [ 5.16214400e+00  2.27721660e+01  1.06516439e+02 ... -8.02302600e+00
  -3.37676200e+00 -3.47322000e-01]
 ...
 [-1.69849600e+00 -2.11304400e+00 -8.02302600e+00 ...  1.32542028e+03
   5.58380708e+02  9.15557480e+01]
 [-8.60152000e-01 -7.60828000e-01 -3.37676200e+00 ...  5.58380708e+02
   6.96391596e+02  1.06598076e+02]
 [-5.91200000e-03  3.17532000e-01 -3.47322000e-01 ...  9.15557480e+01
   1.06598076e+02  1.05604956e+02]]

10000
(100,)
[91.255 90.27  90.716 89.922 90.944 88.477 87.211 87.137 87.519 90.007
 91.827 89.937 90.559 90.244 90.874 89.86  88.283 87.91  87.484 89.826
 89.799 88.769 89.385 89.109 89.381 89.77  88.787 87.379 88.683 89.761
 89.961 89.216 89.657 90.118 90.235 89.685 88.847 87.664 89.397 90.713
 89.595 88.878 89.514 89.769 89.757 88.573 87.517 88.023 88.302 89.494
 89.797 88.788 89.073 89.037 88.798 89.13  88.259 88.089 89.067 88.472
 90.31  89.594 88.04  89.237 88.227 88.688 89.567 87.395 87.611 87.724
 88.892 88.719 88.805 89.847 88.495 89.421 89.732 86.685 86.859 87.242
 88.93  88.47  88.548 88.894 88.506 88.65  88.376 87.667 88.388 87.959
 90.374 89.056 89.319 88.728 88.885 88.219 89.627 88.218 87.803 87.568]
(100, 100)
[[5112.583975 4430.27215  3980.62442  ... 2314.35841  2186.324235
  2149.57416 ]
 [4430.27215  5077.8651   4568.86668  ... 2450.45114  2380.28819
  2346.16764 ]
 [3980.62442  4568.86668  5236.253344 ... 2840.755912 2745.806052
  2665.992312]
 ...
 [2314.35841  2450.45114  2840.755912 ... 5312.420476 4842.782946
  4518.903176]
 [2186.324235 2380.28819  2745.806052 ... 4842.782946 5349.388191
  4873.198896]
 [2149.57416  2346.16764  2665.992312 ... 4518.903176 4873.198896
  5211.649376]]

  '''
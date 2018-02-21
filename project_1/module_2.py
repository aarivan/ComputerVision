import cv2
import numpy as np
import math
import os
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import statsmodels.stats.correlation_tools



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



imageDirFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_faces/"
imageDirNonFaces = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_non_faces/" 
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

for file in os.listdir(imageDirFaces):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDirFaces, file))

for file in os.listdir(imageDirNonFaces):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDirNonFaces, file))


images_grey_scaled = []
# sum_of_images_faces_greyscale = np.zeros(169,np.float)
for img_path in image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (7, 7))
	images_grey_scaled.append(img_scaled)



## Faces Gaussian Mixture
faces_random_list = np.random.permutation(100)

dataset_images = []
for i in faces_random_list:
	im_reshape = images_grey_scaled[i].reshape((1, 49))
	dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

faces_lamda_k = []
faces_mean_k = []
faces_sig_k = []
faces_covariance_epsilon = []

# Initializing values for lambda_k, mean_k and sig_k
for K in range(2, 5):
	# Lambda_k value
	lambda_k = []
	for i in range(0, K):
		lambda_k.append(1/ float(K))

	# Mean_k value
	mean_tuple_list = []
	means_index = np.random.permutation(100)
	for i in range(0, K):
		mean_tuple_list.append(dataset_images[means_index[i]])

	mean_k = np.vstack(tuple(mean_tuple_list))

	# Sig_k values
	dataset_mean = np.mean(dataset_matrix, axis = 0)
	dataset_variance = np.zeros((49, 49))

	for i in range(0, 100):
		data = dataset_matrix[i, :]
		data = data.reshape((49, 1))
		mean_temp = dataset_mean.reshape((49, 1))
		temp = data - mean_temp
		dataset_variance += np.matmul(temp, temp.T)

	dataset_variance = dataset_variance / 100

	sig_k = np.zeros((49, 49, K))
	for i in range(0, K):
		sig_k[:, :, i] = dataset_variance

	## Main Iteration for computations
	iterations = 0
	previous_L = 1000000
	precision = 0.01
	covariance_epsilon = [0.0] * K 
	while True:
		## Expectation Step
		norm = np.zeros((K, 100))
		r = np.zeros((K, 100))

		for i in range(0, K):
			print covariance_epsilon[i]
			if covariance_epsilon[i] > 0.0:
				print "I am here"
				nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
				mvn = multivariate_normal(mean_k[i], nearest_cov) 
			else:
				mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])

			pdf = mvn.pdf(dataset_matrix)
			pdf = lambda_k[i] * pdf
			pdf = pdf.reshape((1, 100))
			norm[i, :] = pdf

		sum_norm = np.sum(norm, axis = 0)

		for i in range(0, 100):
			for k in range(0, K):
				if sum_norm[i] == 0.0:
					r[k, i] = 0.000001
				else:
					r[k, i] = norm[k, i] / sum_norm[i]

		r_sum_all = np.sum(np.sum(r, axis = 0))
		r_sum_rows = np.sum(r, axis = 1)

		for k in range(0, K):
			# Update lambda_k
			lambda_k[k] = r_sum_rows[k] / r_sum_all

			# Update mean_k
			new_mean_k = np.zeros((1, 49))
			for i in range(0, 100):
				x = dataset_matrix[i, :]
				new_mean_k = new_mean_k + r[k][i] * x

			new_mean_k = new_mean_k / r_sum_rows[k]
			mean_k[k, :] = new_mean_k

			# Update sig_k
			new_sig_k = np.zeros((49, 49))
			for i in range(0, 100):
				x = dataset_matrix[i, :].reshape((49, 1))
				temp_mean = mean_k[k, :].reshape((49, 1))
				x = x - temp_mean
				new_sig_k += r[k][i] * np.matmul(x, x.T)

			new_sig_k = new_sig_k / r_sum_rows[k]
			sig_k[:, :, k] = new_sig_k

		temp = np.zeros((K, 100))
		for i in range(0, K):
			diag = np.amax(sig_k[:, :, i])
			# print(math.sqrt(diag))
			covariance_epsilon[i] = math.sqrt(diag) - 0.0001
			nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
			mvn = multivariate_normal(mean_k[i], nearest_cov) 
			pdf = mvn.pdf(dataset_matrix)
			pdf = lambda_k[i] * pdf
			pdf = pdf.reshape((1, 100))
			temp[i, :] = pdf

		# print covariance_epsilon
		# break
		sum_temp = np.sum(temp, axis = 0)
		# print(sum_temp.shape)
		# print(sum_temp)
		# break

		## Get average value for -inf
		avg = 0.0
		ct = 0
		for i in range(0, 100):
			if sum_temp[i] == 0.0:
				avg += 0.0
			else:
				ct += 1
				avg += sum_temp[i]
		avg = avg / ct

		## Replace average values at 0
		for i in range(0, 100):
			if sum_temp[i] == 0.0:
				sum_temp[i] = avg

		temp_log = []
		for i in range(0, 100):
			temp_log.append(np.log(sum_temp[i]))

		# print temp_log
		# break
		L = np.sum(temp_log)

		iterations += 1
		# print previous_L
		# print L
		# print abs(L - previous_L)

		# print
		print("Iterations Completed: ", str(iterations))
		# print

		if abs(L - previous_L) < precision or iterations > 20:
			break
		else:
			previous_L = L

	faces_lamda_k.append(lambda_k)
	faces_mean_k.append(mean_k)
	faces_sig_k.append(sig_k)
	faces_covariance_epsilon.append(covariance_epsilon)


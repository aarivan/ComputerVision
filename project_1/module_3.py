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

def multivariate_t(X, mean, covariance, nu, D, size):
    c = np.exp(special.gammaln((nu + D) / 2)) - special.gammaln(nu / 2)
    c = c / (((nu * math.pi) ** (D / 2)) * (math.sqrt(np.linalg.det(covariance))))

    pdf = np.zeros((1, size))
    temp_mean = mean.reshape((D, 1))
    for i in range(0, size):
        temp_data = X[i, :].reshape((D, 1))
        temp_data_minus_mean = temp_data - temp_mean
        temp_data_minus_mean_transpose_inv_sig = np.matmul(temp_data_minus_mean.T, np.linalg.inv(covariance))
        temp_data_minus_mean_transpose_inv_sig_data_minus_mean = np.matmul(temp_data_minus_mean_transpose_inv_sig, temp_data_minus_mean)
        pdf[0, i] = temp_data_minus_mean_transpose_inv_sig_data_minus_mean

    for i in range(0, size):
        pdf[0, i] = 1 + (pdf[0, i] / nu)
        pdf[0, i] = pdf[0, i] ** (-1 * (nu + D) / 2)
        pdf[0, i] = c * pdf[0, i]
    
    return pdf


def find_mean(matrix):
	return np.mean(matrix, axis = 0)


def  find_variance(face_matrix, face_mean_k, face_variance):
	for i in range(0, 1000):
		data = face_matrix[i, :]
		temp = data - face_mean_k
		temp = temp.reshape((1, 100))
		face_variance += np.matmul(temp.T, temp)
	return face_variance/1000


def find_inverse(matrix):
	return np.linalg.inv(matrix)


def compute_E_hi(nu_k, D, delta_i):
	E_hi = np.zeros((1, 1000))
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



imageDirface = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_faces/"
imageDirNonface = "/home/aarivan/Desktop/computer_vision/projects/ComputerVision/project_1/extracted_data/train_data/train_non_faces/" 

face_image_path_list = []
non_face_image_path_list = []

valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]

for file in os.listdir(imageDirface):
	extension = os.path.splitext(file)[1]
	if extension.lower() not in valid_image_extensions:
		continue
	face_image_path_list.append(os.path.join(imageDirface, file))

for file in os.listdir(imageDirNonface):
	extension = os.path.splitext(file)[1]
	if extension.lower() not in valid_image_extensions:
		continue
	non_face_image_path_list.append(os.path.join(imageDirNonface, file))


face_images_grey_scaled = []
for img_path in face_image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (10, 10))
	face_images_grey_scaled.append(img_scaled)

non_face_images_grey_scaled = []
for img_path in non_face_image_path_list:
	img = cv2.imread(img_path)
	img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
	img_scaled = cv2.resize(img, (10, 10))
	non_face_images_grey_scaled.append(img_scaled)


face_dataset_images = []
non_face_dataset_images = []

for face_image in face_images_grey_scaled:
	im_reshape = face_image.reshape((1, 100))
	face_dataset_images.append(im_reshape)

for non_face_images in non_face_images_grey_scaled:
	im_reshape = non_face_images.reshape((1, 100))
	non_face_dataset_images.append(im_reshape)

face_images_grey_scaled_tupled = tuple(face_dataset_images)
face_images_grey_scaled_matrix = np.vstack(face_images_grey_scaled_tupled)

non_face_images_grey_scaled_tupled = tuple(non_face_dataset_images)
non_face_images_grey_scaled_matrix = np.vstack(non_face_images_grey_scaled_tupled)


face_nu_k_final = []
non_face_nu_k_final = []

face_mean_k_final = []
non_face_mean_k_final = []

face_sig_k_final = []
non_face_sig_k_final = []

face_covariance_epsilon_final = []
non_face_covariance_epsilon_final = []


face_mean_k = face_matrix_mean = find_mean(face_images_grey_scaled_matrix)
non_face_mean_k = non_face_matrix_mean = find_mean(non_face_images_grey_scaled_matrix)

# Initialize Nu to large value
face_nu_k = 10
non_face_nu_k = 10

face_variance = np.zeros((100, 100))
face_sig_k = face_variance = find_variance(face_images_grey_scaled_matrix, face_mean_k, face_variance)

non_face_variance = np.zeros((100, 100))
non_face_sig_k = non_face_variance = find_variance(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_variance)

loop_count = 0
face_previous_L = 1000000
non_face_previous_L = 1000000
face_delta_i = np.zeros((1, 1000))
non_face_delta_i = np.zeros((1, 1000))


while True:
	
	# EXPECTATION STEP

	# Calculating delta_i
	face_sig_k_inverse = find_inverse(face_sig_k)
	non_face_sig_k_inverse = find_inverse(non_face_sig_k)

	for i in range(0, 1000):
		face_data = face_images_grey_scaled_matrix[i, :].reshape((1, 100))
		non_face_data = non_face_images_grey_scaled_matrix[i, :].reshape((1, 100))

		face_mu = face_mean_k.reshape((1, 100))
		non_face_mu = non_face_mean_k.reshape((1, 100))

		face_temp = face_data - face_mu
		non_face_temp = non_face_data - non_face_mu

		face_delta = np.matmul(face_temp, face_sig_k_inverse)
		non_face_delta = np.matmul(non_face_temp, non_face_sig_k_inverse)

		face_delta = np.matmul(face_delta, face_temp.T)
		non_face_delta = np.matmul(non_face_delta, non_face_temp.T)

		face_delta_i[0, i] = face_delta
		non_face_delta_i[0, i] = non_face_delta

	# Computing E[h i]
	face_E_hi = compute_E_hi(face_nu_k, 100, face_delta_i)
	non_face_E_hi = compute_E_hi(non_face_nu_k, 100, non_face_delta_i)

	# Compute the E[log h i]
	face_E_log_hi = np.zeros((1, 1000))
	face_E_log_hi = compute_E_log_hi(face_E_log_hi, face_nu_k, 100, face_delta_i)

	non_face_E_log_hi = np.zeros((1, 1000))
	non_face_E_log_hi = compute_E_log_hi(non_face_E_log_hi, non_face_nu_k, 100, non_face_delta_i)

	## MAXIMIZATION STEP ##

	# Calculating New Mean
	face_E_hi_sum = np.sum(face_E_hi, axis = 1)
	non_face_E_hi_sum = np.sum(non_face_E_hi, axis = 1)

	face_new_mean_k = np.zeros((1, 100))
	non_face_new_mean_k = np.zeros((1, 100))

	face_mean_k = face_new_mean_k = calculate_new_mean(face_images_grey_scaled_matrix, face_mean_k, face_E_hi, face_new_mean_k, face_E_hi_sum, 100)
	non_face_mean_k = non_face_new_mean_k = calculate_new_mean(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_E_hi, non_face_new_mean_k, non_face_E_hi_sum, 100)


	# Calculating New Sigma
	face_new_sig_k = np.zeros((100, 100))
	non_face_new_sig_k = np.zeros((100, 100))

	face_sig_k = face_new_sig_k = calculate_new_sig(face_images_grey_scaled_matrix, face_mean_k, face_E_hi, face_E_hi_sum, face_new_sig_k, 100)
	non_face_sig_k = non_face_new_sig_k = calculate_new_sig(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_E_hi, non_face_E_hi_sum, non_face_new_sig_k, 100)

	face_E_log_hi_sum = np.sum(face_E_log_hi, axis = 1)
	non_face_E_log_hi_sum = np.sum(non_face_E_log_hi, axis = 1)


	# Calculating New Nu
	face_nu_k = optimize.fminbound(t_cost, 0, 10000, args=(face_E_hi, face_E_hi_sum, face_E_log_hi, face_E_log_hi_sum, 1000))
	non_face_nu_k = optimize.fminbound(t_cost, 0, 10000, args=(non_face_E_hi, non_face_E_hi_sum, non_face_E_log_hi, non_face_E_log_hi_sum, 1000))
	

	# Calculating Log likelihood
	face_new_delta_i = np.zeros((1, 1000))
	non_face_new_delta_i = np.zeros((1, 1000))


	face_sig_k_inverse = find_inverse(face_sig_k)
	non_face_sig_k_inverse = find_inverse(non_face_sig_k)

	face_delta_i = face_new_delta_i = calculate_delta_i(face_images_grey_scaled_matrix, face_mean_k, face_sig_k_inverse, face_new_delta_i, 100)
	non_face_delta_i = non_face_new_delta_i = calculate_delta_i(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_sig_k_inverse, non_face_new_delta_i, 100)


	# Calculating value of 'L'
	size = 1000
	(face_sig_sign, face_sig_k_logdet) = calculate_slogdet(face_sig_k)
	(non_face_sig_sign, non_face_sig_k_logdet) = calculate_slogdet(non_face_sig_k)

	face_sig_k_logdet_half = face_sig_k_logdet / 2
	non_face_sig_k_logdet_half = non_face_sig_k_logdet / 2

	face_gammaln_nu_D_half = calculate_gammaln(face_nu_k + 100)
	non_face_gammaln_nu_D_half = calculate_gammaln(non_face_nu_k + 100)


	# face_log_nu_pi = (100 / 2) * np.log(face_nu_k * math.pi)
	face_log_nu_pi = calculate_log_nu_pi(100, face_nu_k)
	# non_face_log_nu_pi = (100 / 2) * np.log(non-face_nu_k * math.pi)
	non_face_log_nu_pi = calculate_log_nu_pi(100, non_face_nu_k)

	face_gammaln_nu_half = calculate_gammaln(face_nu_k)
	non_face_gammaln_nu_half = calculate_gammaln(non_face_nu_k)

	
	face_L = size * (face_gammaln_nu_D_half - face_log_nu_pi - face_sig_k_logdet_half - face_gammaln_nu_half)
	non_face_L = size * (non_face_gammaln_nu_D_half - non_face_log_nu_pi - non_face_sig_k_logdet_half - non_face_gammaln_nu_half)

	face_log_delta_nu_sum, non_face_log_delta_nu_sum = 0, 0

	face_log_delta_nu_sum = calculate_log_delta_nu_sum(face_delta_i, face_nu_k, face_log_delta_nu_sum)
	non_face_log_delta_nu_sum = calculate_log_delta_nu_sum(non_face_delta_i, non_face_nu_k, non_face_log_delta_nu_sum)

	face_log_delta_nu_sum /= 2
	non_face_log_delta_nu_sum /= 2


	face_L -= ((face_nu_k + 100) * face_log_delta_nu_sum)
	non_face_L -= ((non_face_nu_k + 100) * non_face_log_delta_nu_sum)

	loop_count += 1
	
	print "Iterations Completed: " + str(loop_count)
	print

	if abs(face_L - face_previous_L) > 0.01:
		face_previous_L = face_L
	if abs(face_L - face_previous_L) > 0.01:
		non_face_previous_L = non_face_L
	if loop_count > 100:
		break

face_nu_k_final = face_nu_k
non_face_nu_k_final = non_face_nu_k

face_mean_k_final = face_mean_k
non_face_mean_k_final = non_face_mean_k

face_sig_k_final = face_sig_k
non_face_sig_k_final = non_face_sig_k

# print face_nu_k_final
# print face_mean_k_final.shape
# print face_mean_k_final
# print face_sig_k_final.shape
# print face_sig_k_final
# print
# print non_face_nu_k_final
# print non_face_mean_k_final.shape
# print non_face_mean_k_final
# print non_face_sig_k_final.shape
# print non_face_sig_k_final

print multivariate_t(non_face_images_grey_scaled_matrix, non_face_mean_k_final, non_face_sig_k_final, non_face_nu_k_final, 100, 1000)

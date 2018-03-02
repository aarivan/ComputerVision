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


def find_mean(matrix):
    return np.mean(matrix, axis = 0)


def  find_variance(matrix, mean_k, variance):
    for i in range(0, 1000):
        data = matrix[i, :].reshape((100, 1))
        temp = data - mean_k.reshape((100, 1))
        # temp = temp.reshape((1, 100))
        variance += np.matmul(temp, temp.T)
    return variance/1000


def calculate_E_hi_sub_calculation(phi, inv_sig):
    phi_transpose = phi.T
    phi_transpose_times_sig_inv = np.matmul(phi_transpose, inv_sig)
    
    product_matrix = np.matmul(phi_transpose_times_sig_inv, phi)
    identity_matrix = np.identity(4)

    return np.linalg.inv(product_matrix + identity_matrix)


def calculate_E_hi(phi, inv_sig, matrix, matrix_mean, E_hi):
    phi_transpose = phi.T
    phi_transpose_times_sig_inv = np.matmul(phi_transpose, inv_sig)
    
    temp_mean = matrix_mean.reshape((100, 1))
    temp_inv = calculate_E_hi_sub_calculation(phi, inv_sig)
    
    for i in range(0, 1000):
        data = matrix[i, :].reshape((100, 1))
        data_mean = data - temp_mean
        temp = np.matmul(temp_inv, phi_transpose_times_sig_inv)
        temp = np.matmul(temp, data_mean)
        E_hi[i, :] = temp.T
    return E_hi


def calculate_E_hi_Ehi_T(faces_phi, faces_inv_sig, E_hi, E_hi_hi_T):
    faces_temp_inv = calculate_E_hi_sub_calculation(faces_phi, faces_inv_sig) 
    for i in range(0, 1000):
        temp_E_hi = E_hi[i, :].reshape((4, 1))
        temp = np.matmul(temp_E_hi, temp_E_hi.T)
        temp += faces_temp_inv 
        E_hi_hi_T[:, :, i] = temp

    return E_hi_hi_T


def calculate_E_hi_hi_T_sum_inv(E_hi_hi_T_sum_inv, E_hi_hi_T):
    for i in range(0, 1000):
        E_hi_hi_T_sum_inv += E_hi_hi_T[:, :, i]
    E_hi_hi_T_sum_inv = np.linalg.inv(E_hi_hi_T_sum_inv)
    return E_hi_hi_T_sum_inv


def calculate_new_phi(matrix, matrix_mean, E_hi, phi_new):
    for i in range(0, 1000):
        data = matrix[i, :].reshape((100, 1))
        temp_mean = matrix_mean.reshape((100, 1))
        data_mean = data - temp_mean
        temp_E_hi = E_hi[i, :].reshape((1, 4))
        phi_new += np.matmul(data_mean, temp_E_hi)
    return phi_new


def calculate_new_sigma(matrix, matrix_mean, E_hi, phi, sig_new):
    for i in range(0, 1000):
        data = matrix[i, :].reshape((100, 1))
        temp_mean = matrix_mean.reshape((100, 1))
        x_mean = data - temp_mean
        prod_x_mean = np.matmul(x_mean, x_mean.T)
        temp_E_h_i = E_hi[i, :].reshape((4, 1))
        temp_phi_E_h_i = np.matmul(phi, temp_E_h_i)
        temp_phi_E_h_i_x_mean_transpose = np.matmul(temp_phi_E_h_i, x_mean.T)
        sig_new += prod_x_mean - temp_phi_E_h_i_x_mean_transpose
    sig_new = sig_new.diagonal().reshape((1, 100))
    sig_new = sig_new / 1000
    return sig_new


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


faces_matrix_mean = find_mean(faces_images_grey_scaled_matrix)
non_faces_matrix_mean = find_mean(non_faces_images_grey_scaled_matrix)

faces_matrix_mean = faces_matrix_mean.reshape((1, 100))
non_faces_matrix_mean = non_faces_matrix_mean.reshape((1, 100))



# Initialize Phi
np.random.seed(0)
# phi = np.random.randn(100, 4)
faces_phi = np.random.randn(100, 4)
non_faces_phi = np.random.randn(100, 4)

faces_sig = np.zeros((100, 100))
non_faces_sig = np.zeros((100, 100))

faces_sig = find_variance(faces_images_grey_scaled_matrix, faces_matrix_mean, faces_sig)
non_faces_sig = find_variance(non_faces_images_grey_scaled_matrix, non_faces_matrix_mean, non_faces_sig)

faces_sig = np.square(faces_sig)
non_faces_sig = np.square(non_faces_sig)

faces_sig = np.sum(faces_sig, axis = 0)
non_faces_sig = np.sum(non_faces_sig, axis = 0)

faces_sig /= 1000
non_faces_sig /= 1000

faces_sig = faces_sig.reshape((1, 100))
non_faces_sig = non_faces_sig.reshape((1, 100))

iterations = 0
previous_L = 1000000
precision = 0.01

while True:

    # EXPECTATION STEP

    faces_divide_array = np.divide(1, faces_sig)
    non_faces_divide_array = np.divide(1, non_faces_sig)

    faces_inv_sig = np.diag(faces_divide_array[0])
    non_faces_inv_sig = np.diag(non_faces_divide_array[0])

    # Calculating E_hi    
    faces_E_hi = np.zeros((1000, 4))
    non_faces_E_hi = np.zeros((1000, 4))

    faces_E_hi = calculate_E_hi(faces_phi, faces_inv_sig, faces_images_grey_scaled_matrix, faces_matrix_mean, faces_E_hi)
    non_faces_E_hi = calculate_E_hi(non_faces_phi, non_faces_inv_sig, non_faces_images_grey_scaled_matrix, non_faces_matrix_mean, non_faces_E_hi)

    # Calculating E_hi_hi_T
    faces_E_hi_hi_T = np.zeros((4, 4, 1000))
    faces_E_hi_hi_T = calculate_E_hi_Ehi_T(faces_phi, faces_inv_sig, faces_E_hi, faces_E_hi_hi_T)

    non_faces_E_hi_hi_T = np.zeros((4, 4, 1000))
    non_faces_E_hi_hi_T = calculate_E_hi_Ehi_T(non_faces_phi, non_faces_inv_sig, non_faces_E_hi, non_faces_E_hi_hi_T)
    

    ## MAXIMIZATION STEP ##

    faces_E_hi_hi_T_sum_inv = np.zeros((4, 4))
    faces_E_hi_hi_T_sum_inv = calculate_E_hi_hi_T_sum_inv(faces_E_hi_hi_T_sum_inv, faces_E_hi_hi_T)

    non_faces_E_hi_hi_T_sum_inv = np.zeros((4, 4))
    non_faces_E_hi_hi_T_sum_inv = calculate_E_hi_hi_T_sum_inv(faces_E_hi_hi_T_sum_inv, faces_E_hi_hi_T)

    # Calculating new phi 
    faces_phi_new = np.zeros((100, 4))
    faces_phi_new = calculate_new_phi(faces_images_grey_scaled_matrix, faces_matrix_mean, faces_E_hi, faces_phi_new)
    faces_phi_new = np.matmul(faces_phi_new, faces_E_hi_hi_T_sum_inv)
    faces_phi = faces_phi_new

    non_faces_phi_new = np.zeros((100, 4))
    non_faces_phi_new = calculate_new_phi(faces_images_grey_scaled_matrix, faces_matrix_mean, faces_E_hi, faces_phi_new)
    non_faces_phi_new = np.matmul(non_faces_phi_new, non_faces_E_hi_hi_T_sum_inv)
    non_faces_phi = non_faces_phi_new
    
    # Calculating new sigma
    faces_sig_new = np.zeros((100, 100))
    faces_sig = faces_sig_new = calculate_new_sigma(faces_images_grey_scaled_matrix, faces_matrix_mean, faces_E_hi, faces_phi, faces_sig_new)

    non_faces_sig_new = np.zeros((100, 100))
    non_faces_sig = non_faces_sig_new = calculate_new_sigma(non_faces_images_grey_scaled_matrix, non_faces_matrix_mean, non_faces_E_hi, non_faces_phi, non_faces_sig_new)
    

    # Calculating Diagonal Covariance
    faces_sig_diag = np.diag(faces_sig[0])
    non_faces_sig_diag = np.diag(non_faces_sig[0])

    ## Computing data log likelihood
    faces_mvn_cov = np.matmul(faces_phi, faces_phi.T) 
    faces_mvn_cov += faces_sig_diag
    ##### ------------- yet to handle positive semi-definite error ----------- #####
    faces_mvn = multivariate_normal(faces_matrix_mean[0], faces_mvn_cov)
    faces_pdf = faces_mvn.pdf(faces_images_grey_scaled_matrix)

    non_faces_mvn_cov = np.matmul(non_faces_phi, non_faces_phi.T) 
    non_faces_mvn_cov += non_faces_sig_diag
    ##### ------------- yet to handle positive semi-definite error ----------- #####
    non_faces_mvn = multivariate_normal(non_faces_matrix_mean[0], non_faces_mvn_cov)
    non_faces_pdf = non_faces_mvn.pdf(non_faces_images_grey_scaled_matrix)

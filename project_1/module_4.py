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


def  find_variance(matrix, mean_k):
    variance = np.zeros((100, 100))
    for i in range(0, 1000):
        data = matrix[i, :].reshape((100, 1))
        temp = data - mean_k.reshape((100, 1))
        # temp = temp.reshape((1, 100))
        variance += np.matmul(temp, temp.T)
    return variance/1000

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


def calculate_mean_k(mean_k, images_grey_scaled_matrix, means_index, k): 
    mean_k = np.zeros((100, k))
    for i in range(0, k):
        mean_k[:, i] = images_grey_scaled_matrix[means_index[i], :].reshape(100)
    return mean_k

def calculate_variance(images_grey_scaled_matrix, mean_k):
    variance = np.zeros((100, 100))
    for i in range(0, 1000):
        data = images_grey_scaled_matrix[i, :].reshape((1, 100))
        mu = mean_k[:, j].reshape((1, 100))
        temp = data - mu
        temp = temp.reshape((1, 100))
        variance += np.matmul(temp.T, temp)
    return variance

def calculate_mixture_t_cost_func(nu, D, posterior_matrix, E_h_i, E_log_h_i, k, size):
    first_term = -1 * special.psi(nu / 2)
    second_term =  np.log(nu / 2)
    third_term = 1
    fourth_term = 0.0
    fourth_term_denominator = 0.0
    for i in range(0, size):
        fourth_term_denominator += posterior_matrix[i, k]
        E_log_h_i_minus_E_h_i = E_log_h_i[k, i] - E_h_i[k, i]
        fourth_term += posterior_matrix[i, k] * E_log_h_i_minus_E_h_i
    fourth_term /= fourth_term_denominator
    fifth_term = special.psi((nu + D) / 2)
    sixth_term = -1 * np.log((nu + D) / 2)
    # print first_term, second_term, third_term, fourth_term, fifth_term, sixth_term
    val = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term
    return val


def calculate_delta_i(face_sig_k, face_images_grey_scaled_matrix, face_mean_k):
    face_delta_i = np.zeros((k, 1000))
    for j in range(0, k):
        face_sig_k_inverse = np.linalg.inv(face_sig_k[:, :, j])
        for i in range(0, 1000):
            x = face_images_grey_scaled_matrix[i, :].reshape((1, 100))
            mu = face_mean_k[:, j].reshape((1, 100))
            temp = x - mu
            temp_delta = np.matmul(temp, face_sig_k_inverse)
            temp_delta = np.matmul(temp_delta, temp.T)
            face_delta_i[j, i] = temp_delta
    return face_delta_i


def calculate_E_hi(nu_k, delta_i):
    E_hi = np.zeros((k, 1000))
    for j in range(0, k):
        for i in range(0, 1000):
            E_hi[j, i] = (nu_k[j] + 100) / (nu_k[j] + delta_i[j, i])
    return E_hi

def calculate_E_log_hi(nu_k, delta_i):
    E_log_hi = np.zeros((k, 1000))
    for j in range(0, k):
        for i in range(0, 1000):
            tmp = special.psi((nu_k[j] + 100) / 2) 
            tmp = tmp - np.log(nu_k[j] / 2 + delta_i[j, i] / 2)
            E_log_hi[j, i] = tmp
    return E_log_hi


def calculate_posterior_matrix(images_grey_scaled_matrix, mean_k, sig_k, nu_k, lambda_k):
    posterior_matrix = np.zeros((1000, k))
    for i in range(0, 1000):
        data_entry = images_grey_scaled_matrix[i, :].reshape((1, 100))
        for j in range(0, k):
            posterior_val = multivariate_t(data_entry, mean_k[:, j], sig_k[:, :, j], nu_k[j], 100, 1)
            posterior_matrix[i, j] = lambda_k[j] * posterior_val[0]
    return posterior_matrix


def calculate_face_sum_denominator(images_grey_scaled_matrix, posterior_matrix, E_hi, new_mean_k):
    sum_denominator = 0.0
    for i in range(0, 1000):
        temp_data = images_grey_scaled_matrix[i, :].reshape((100, 1))
        temp_prod = posterior_matrix[i, j] * E_hi[j, i]
        temp = temp_prod * temp_data
        new_mean_k += temp
        sum_denominator += temp_prod
    return sum_denominator

def calculate_face_sum_denominator_sig(face_images_grey_scaled_matrix, face_mean_k, face_posterior_matrix, face_E_hi, face_new_sig_k):
    face_sum_denominator = 0.0
    for i in range(0, 1000):
        temp_data = face_images_grey_scaled_matrix[i, :].reshape((100, 1))
        temp_mean = face_mean_k[:, j].reshape((100, 1))
        temp_data_minus_mean = temp_data - temp_mean
        temp_prod = face_posterior_matrix[i, j] * face_E_hi[j, i]
        temp = np.matmul(temp_data_minus_mean, temp_data_minus_mean.T)
        face_new_sig_k += temp_prod * temp
        face_sum_denominator += temp_prod
    return face_sum_denominator

    
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


face_lambda_k_final = []
non_face_lambda_k_final = []

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


K = 5
for k in range(2, K + 1):
    
    # Initializing the teta parameters
    # Initializing the Mean

    face_means_index = np.random.permutation(1000)
    face_mean_k = calculate_mean_k(face_mean_k, face_images_grey_scaled_matrix, face_means_index, k)

    non_face_means_index = np.random.permutation(1000)
    non_face_mean_k = calculate_mean_k(non_face_mean_k, non_face_images_grey_scaled_matrix, non_face_means_index, k)


    # Initialing Nu - to very large value
    face_nu_k = [10] * k
    non_face_nu_k = [10] * k

    # Initializing Sigma
    face_sig_k = np.zeros((100, 100, k))
    non_face_sig_k = np.zeros((100, 100, k))

    for j in range(0, k):
        face_variance = calculate_variance(face_images_grey_scaled_matrix, face_mean_k)
        face_variance /= 1000
        face_sig_k[:, :, j] = face_variance

        non_face_variance = calculate_variance(non_face_images_grey_scaled_matrix, non_face_mean_k)
        non_face_variance /= 1000
        non_face_sig_k[:, :, j] = non_face_variance
        

    # Initializing lambda_k
    face_lambda_k = [1 / float(k)] * k
    non_face_lambda_k = [1 / float(k)] * k

    precision = 0.01
    loop_count = 0
    previous_L = 1000000
    print k

    while True:
        ## Expectation Step
        # Computing Delta's
        face_delta_i = calculate_delta_i(face_sig_k, face_images_grey_scaled_matrix, face_mean_k)
        non_face_delta_i = calculate_delta_i(non_face_sig_k, non_face_images_grey_scaled_matrix, non_face_mean_k)


        # Compute E[h i]'s
        face_E_hi = calculate_E_hi(face_nu_k, face_delta_i)
        non_face_E_hi = calculate_E_hi(non_face_nu_k, non_face_delta_i)



        # Compute the E[log h i]
        face_E_log_hi = calculate_E_log_hi(face_nu_k, face_delta_i)
        non_face_E_log_hi = calculate_E_log_hi(non_face_nu_k, non_face_delta_i)


        # Compute the l_posterior matrix
        face_posterior_matrix = calculate_posterior_matrix(face_images_grey_scaled_matrix, face_mean_k, face_sig_k,face_nu_k, face_lambda_k)
        non_face_posterior_matrix = calculate_posterior_matrix(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_sig_k, non_face_nu_k, non_face_lambda_k)
        
        
        ## Maximization Step
        # Compute the new lambda_k's
        face_posterior_matrix_colwise_sum = np.sum(face_posterior_matrix, axis = 0)
        non_face_posterior_matrix_colwise_sum = np.sum(non_face_posterior_matrix, axis = 0)

        face_total_posterior_matrix_sum = np.sum(face_posterior_matrix)
        non_face_total_posterior_matrix_sum = np.sum(non_face_posterior_matrix)
        
        for j in range(0, k):
            face_lambda_k[j] = face_posterior_matrix_colwise_sum[j] / face_total_posterior_matrix_sum
            non_face_lambda_k[j] = non_face_posterior_matrix_colwise_sum[j] / non_face_total_posterior_matrix_sum



        # Compute new mean_k's
        for j in range(0, k):
            face_new_mean_k = np.zeros((100, 1))
            face_sum_denominator = calculate_face_sum_denominator(face_images_grey_scaled_matrix, face_posterior_matrix, face_E_hi, face_new_mean_k)
            face_new_mean_k /= face_sum_denominator
            face_mean_k[:, j] = face_new_mean_k.reshape(100)

            non_face_new_mean_k = np.zeros((100, 1))
            non_face_sum_denominator = calculate_face_sum_denominator(non_face_images_grey_scaled_matrix, non_face_posterior_matrix, non_face_E_hi, non_face_new_mean_k)
            non_face_new_mean_k /= non_face_sum_denominator
            non_face_mean_k[:, j] = non_face_new_mean_k.reshape(100)
        


        # Compute new sig_k's
        for j in range(0, k):
            face_new_sig_k = np.zeros((100, 100))
            face_sum_denominator = calculate_face_sum_denominator_sig(face_images_grey_scaled_matrix, face_mean_k, face_posterior_matrix, face_E_hi, face_new_sig_k)
            face_new_sig_k /= face_sum_denominator
            face_sig_k[:, :, j] = face_new_sig_k

            non_face_new_sig_k = np.zeros((100, 100))
            non_face_sum_denominator = calculate_face_sum_denominator_sig(non_face_images_grey_scaled_matrix, non_face_mean_k, non_face_posterior_matrix, non_face_E_hi, non_face_new_sig_k)
            non_face_new_sig_k /= non_face_sum_denominator
            non_face_sig_k[:, :, j] = non_face_new_sig_k

        # Compute nu_k's
        for j in range(0, k):
            face_nu_t = optimize.fminbound(calculate_mixture_t_cost_func, 0, 10, args=(100, face_posterior_matrix, face_E_hi, face_E_log_hi, j, 1000))
            face_nu_k[j] = face_nu_t

            non_face_nu_t = optimize.fminbound(calculate_mixture_t_cost_func, 0, 10, args=(100, non_face_posterior_matrix, non_face_E_hi, non_face_E_log_hi, j, 1000))
            non_face_nu_k[j] = non_face_nu_t



        loop_count += 1
        if loop_count >= 1:
            break
    
    face_lambda_k_final.append(face_lambda_k)
    non_face_lambda_k_final.append(non_face_lambda_k)
    
    face_mean_k_final.append(face_mean_k)
    non_face_mean_k_final.append(non_face_mean_k)
    
    face_sig_k_final.append(face_sig_k)
    non_face_sig_k_final.append(non_face_sig_k)
    
    face_nu_k_final.append(face_nu_k)
    non_face_nu_k_final.append(non_face_nu_k)

print face_lambda_k_final
print face_nu_k_final
print face_mean_k_final
print face_sig_k_final
print
print non_face_lambda_k_final
print non_face_nu_k_final
print non_face_mean_k_final
print non_face_sig_k_final
import numpy as np
import lib
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimage


####################################################################################################


def load_images(path: str) -> list:
    """
    Load all images in path

    :param path: path of directory containing image files

    :return images: list of images (each image as numpy.ndarray and dtype=float64)
    """
    imgPaths = glob.glob(path)
    imgPaths = sorted(imgPaths)

    imgArray = []

    for index, name in enumerate(imgPaths):
        imgArray.append(mpimage.imread(name))

    return imgArray


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    :param images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    :return D: data matrix that contains the flattened images as rows
    """
    imgArray = []

    for img in (images):
        imgArray.append(img.ravel().astype(float))

    return imgArray


def calculate_svd(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform SVD analysis for given data matrix.

    :param D: data matrix of size n x m where n is the number of observations and m the number of variables

    :return eigenvec: matrix containing principal components as rows
    :return singular_values: singular values associated with eigenvectors
    :return mean_data: mean that was subtracted from data
    """
    ...

    mean_data = np.mean(D, axis=0)

    D -= mean_data

    U, S, V = np.linalg.svd(
        D, full_matrices=False)

    print(U.shape, S.shape, V.shape)

    return V, S, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    :param singular_values: vector containing singular values
    :param threshold: threshold for determining k (default = 0.8)

    :return k: threshold index
    """
    ...
    norm_singular_values = singular_values / np.sum(singular_values)

    k = 0
    index = 0
    while k <= threshold:
        k += norm_singular_values[index]
        index += 1

    return index


def project_faces(pcs: np.ndarray, mean_data: np.ndarray, images: list) -> np.ndarray:
    """
    Project given image set into basis.

    :param pcs: matrix containing principal components / eigenfunctions as rows
    :param images: original input images from which pcs were created
    :param mean_data: mean data that was subtracted before computation of SVD/PCA

    :return coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """
    print('\n PCS: ', pcs.shape)
    print('\n MEAN: ', mean_data.shape)
    print('\n Images: ', images)

    ...


def identify_faces(coeffs_train: np.ndarray, coeffs_test: np.ndarray) -> (
        np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    :param coeffs_train: coefficients for training images, each image is represented in a row
    :param coeffs_test: coefficients for test images, each image is represented in a row

    :return indices: array containing the indices of the found matches
    """
    ...


if __name__ == '__main__':
    ...
    imgArray = load_images("data/train/*****.png")
    imgDimensions = np.shape(imgArray[0])

    imgMatrix = setup_data_matrix(imgArray)

    eigenvec, sv, mean = calculate_svd(imgMatrix)

    lib.visualize_eigenfaces(n=10, sv=sv, pcs=eigenvec,
                             dim_x=imgDimensions[0], dim_y=imgDimensions[1])

    index = accumulated_energy(sv)
    lib.plot_singular_values_and_energy(sv, index)

    project_faces(sv[0:index], mean, imgMatrix)
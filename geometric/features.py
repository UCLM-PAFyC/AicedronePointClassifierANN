"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import log as ln

import math
import sys


class GeometricFeatures:
    """
    Contains all the methods for calculating the geometric features
    of a neighborhood of a 3D point cloud
    """

    @staticmethod
    def sum_of_eigenvalues(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\sum_{i} \lambda_{i}$
        """
        if eigenvalues is None:
            return float('NaN')

        return eigenvalues[0] + eigenvalues[1] + eigenvalues[2]


    @staticmethod
    def omnivariance(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\left( \prod_{i} \lambda_{i} \right)^\frac{1}{3}$
        """
        if eigenvalues is None:
            return float('NaN')

        product: float = eigenvalues[0] * eigenvalues[1] * eigenvalues[2]
        if product < 0:
            return float('NaN')

        return math.pow(product, 1.0 / 3.0)


    @staticmethod
    def eigenentropy(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $-\sum_{i}\lambda_{i}\ln\left( \lambda_{i} \right )$
        """
        if eigenvalues is None:
            return float('NaN')

        # If lambda = 0, limit indetermination 0 * \inf
        sum: float = 0
        for eigenvalue in eigenvalues:
            if eigenvalue <= 0:
                return float('NaN')
            sum += eigenvalue * ln(eigenvalue)
        return -sum


    @staticmethod
    def anisotropy(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{1} - \lambda_{3}}{\lambda_{1}}$
        """
        if eigenvalues is None:
            return float('NaN')

        return (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def linearity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{1} - \lambda_{2}}{\lambda_{1}}$
        """
        if eigenvalues is None:
            return float('NaN')

        return (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def planarity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{2} - \lambda_{3}}{\lambda_{1}}$
        """
        if eigenvalues is None:
            return float('NaN')

        return (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def sphericity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{3}}{\lambda_{1}}$
        """
        if eigenvalues is None:
            return float('NaN')

        return eigenvalues[2] / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def PCA1(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{1}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        if eigenvalues is None:
            return float('NaN')

        eigenvaluesSum = GeometricFeatures.sum_of_eigenvalues(eigenvalues)
        return eigenvalues[0] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def PCA2(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{2}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        if eigenvalues is None:
            return float('NaN')

        eigenvaluesSum = GeometricFeatures.sum_of_eigenvalues(eigenvalues)
        return eigenvalues[1] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def surface_variation(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{3}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        if eigenvalues is None:
            return float('NaN')

        eigenvaluesSum = GeometricFeatures.sum_of_eigenvalues(eigenvalues)
        return eigenvalues[2] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')


    @staticmethod
    def verticality(eigenvectors: np.array) -> float:
        """
        :param eigenvectors: Eigenvectors associated to the eigenvalues of the covariance matrix of a neighborhood
        :return: $1 - \left | n_{z} \right |$
        """
        if eigenvectors is None:
            return float('NaN')

        z = [0, 0, 1]
        e3 = eigenvectors[2]
        return 1.0 - math.fabs(np.dot(z, e3))


    @staticmethod
    def eigenvalue1(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{1}$
        """
        if eigenvalues is None:
            return float('NaN')

        return eigenvalues[0]


    @staticmethod
    def eigenvalue2(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{2}$
        """
        if eigenvalues is None:
            return float('NaN')

        return eigenvalues[1]


    @staticmethod
    def eigenvalue3(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{3}$
        """
        if eigenvalues is None:
            return float('NaN')

        return eigenvalues[2]
    

class TopologicalFeatures:
    """
    Contains all the methods for calculating the topology features
    of a neighborhood of a 3D point cloud
    """
    @staticmethod
    def height_above(point: np.array, plane: np.array):
        h: float = - plane[3] / plane[2]
        return h - point[2]
    
    
    @staticmethod
    def height_below(point: np.array, plane: np.array):
        h: float = - plane[3] / plane[2]
        return point[2] - h
    
    
    @staticmethod
    def distance_to_plane(point: np.array, plane: np.array):
        numerator = np.abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])
        denominator = np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
        return numerator / denominator
    
    
class RadiometricFeatures:
    """
    Contains all the methods for calculating the radiometric features
    of a neighborhood of a 3D point cloud
    """
    pass
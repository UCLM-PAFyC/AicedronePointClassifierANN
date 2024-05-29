"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import linalg as LA

from scipy.spatial import KDTree

import laspy

from geometric.features import GeometricFeatures
from geometric.features import TopologicalFeatures

from sklearn.linear_model import RANSACRegressor


SCALAR_FIELDS: list[str] = ['Sum_of_eigenvalues', 'Omnivariance', 'Eigenentropy',
                            'Anisotropy', 'Linearity', 'Planarity', 'Sphericity',
                            'PCA1', 'PCA2', 'Surface_variation', 'Verticality',
                            'Eigenvalue1', 'Eigenvalue2', 'Eigenvalue3', 'Height_above',
                            'Height_below', 'Distance_to_plane'
                            ]

MIN_NEIGHBORHOOD: int = 3


def add_scalar_fields(las: laspy.LasData) -> None:
    """
    :param las: LAS file
    :return:
    """
    for scalar_field in SCALAR_FIELDS:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name=scalar_field,
            type=np.float64,
        ))


def get_neighboorhood_data(neighborhood: np.array) -> np.array:
    """
    :param neighborhood: an array of points of the neighborhood of the current point
    :return: Matrix with rows X, Y and Z where each row corresponds to a vector of the components x, y, z of each point respectively
    """
    # Separate x, y and z of each neighborhood in 3 different vectors
    X: list[float] = []
    Y: list[float] = []
    Z: list[float] = []

    # Compute neighborhood matrix
    for neighbor in neighborhood:
        X.append(neighbor[0])
        Y.append(neighbor[1])
        Z.append(neighbor[2])

    # Return matrix
    return np.array([X, Y, Z])


def eigen(point: list[float], kdtree: KDTree, pointcloud: np.array, radius: float, BIAS=False) -> tuple[any, any]:
    """
    :param point: Point at which the neighborhood will be calculated
    :param kdtree: KDTree of the pointcloud previously calculated
    :param pointcloud: The point cloud
    :param radius: Radius of the neighborhood of point
    :param BIAS: BIAS = True means computing the population covariance matrix and BIAS = False, the sample covariance matrix
    :return: Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of a radius 'radius' of the point 'point'
    """
    # Find neighborhood
    idx = kdtree.query_ball_point(point, r=radius)
    neighborhood: np.array = pointcloud[idx]

    if len(neighborhood) < MIN_NEIGHBORHOOD:
        return None, None

    # Compute covariance matrix of the neighborhood
    data = get_neighboorhood_data(neighborhood)
    covarianceMatrix = np.cov(data, bias=BIAS)

    # Return eigenvalues and eigenvectors
    return LA.eig(covarianceMatrix)


def compute_geometric_features(las: laspy.LasData, eigenvalues: np.array, eigenvectors: np.array, idx: int) -> None:
    """
    :param las: LAS file
    :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
    :param eigenvectors: Eigenvectors associated to the eigenvalues of the covariance matrix of a neighborhood
    :param idx: scalar_field index
    :return:
    """
    las[SCALAR_FIELDS[0]][idx] = GeometricFeatures.sum_of_eigenvalues(eigenvalues)
    las[SCALAR_FIELDS[1]][idx] = GeometricFeatures.omnivariance(eigenvalues)
    las[SCALAR_FIELDS[2]][idx] = GeometricFeatures.eigenentropy(eigenvalues)

    las[SCALAR_FIELDS[3]][idx] = GeometricFeatures.anisotropy(eigenvalues)
    las[SCALAR_FIELDS[4]][idx] = GeometricFeatures.linearity(eigenvalues)
    las[SCALAR_FIELDS[5]][idx] = GeometricFeatures.planarity(eigenvalues)
    las[SCALAR_FIELDS[6]][idx] = GeometricFeatures.sphericity(eigenvalues)

    las[SCALAR_FIELDS[7]][idx] = GeometricFeatures.PCA1(eigenvalues)
    las[SCALAR_FIELDS[8]][idx] = GeometricFeatures.PCA2(eigenvalues)
    las[SCALAR_FIELDS[9]][idx] = GeometricFeatures.surface_variation(eigenvalues)

    las[SCALAR_FIELDS[10]][idx] = GeometricFeatures.verticality(eigenvectors)

    las[SCALAR_FIELDS[11]][idx] = GeometricFeatures.eigenvalue1(eigenvalues)
    las[SCALAR_FIELDS[12]][idx] = GeometricFeatures.eigenvalue2(eigenvalues)
    las[SCALAR_FIELDS[13]][idx] = GeometricFeatures.eigenvalue3(eigenvalues)

   
def compute_topological_features(las: laspy.LasData, point: np.array, plane: np.array, idx: int) -> None:
    """
    :param las: LAS file
    :param point: Point of the pointcloud
    :param plane: Plane equation coefficients
    :param idx: scalar_field index
    """
    las[SCALAR_FIELDS[14]][idx] = TopologicalFeatures.height_above(point, plane)
    las[SCALAR_FIELDS[15]][idx] = TopologicalFeatures.height_below(point, plane)
    las[SCALAR_FIELDS[16]][idx] = TopologicalFeatures.distance_to_plane(point, plane)
     
    
def calculate_features(las: laspy.LasData, radius: float, bias: bool = False, callback=None) -> None:
    """
    :param las: Opened las file
    :param radius: Neighborhood radius
    :param bias: Set bias to false to compute the sample covariance or set it to true to compute the population covariance
    :param callback: [OPTIONAL] calls a callback for dealing with the percentage of the computation
    :return: None
    """
    
    add_scalar_fields(las)

    num_points: int = len(las.points)

    # Build kdtree
    pointcloud = np.array(las.xyz)
    T = KDTree(pointcloud)
    
    # Reference plane for topological features
    ransac = RANSACRegressor()
    ransac.fit(pointcloud[:, :2], pointcloud[:, 2])
    
    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_

    plane = np.array([a, b, c, d])

    # Calculate the geometric features of each point in a neighborhood of radius r
    previous_percentage: int = -1
    point_index: int = 0
    for p in pointcloud:
        
        # Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of p
        eigenvalues, eigenvectors = eigen(p, T, pointcloud, radius, bias)

        # Order eigenvalues and associated eigenvectors
        if eigenvalues is not None and eigenvectors is not None:
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Compute features
        compute_geometric_features(las, eigenvalues, eigenvectors, point_index)
        compute_topological_features(las, p, plane, point_index)
        point_index += 1

        # Compute optional percentage
        if callback is not None:
            percentage = max(int(100 * float(point_index) / num_points) - 1, 0)
            if percentage != previous_percentage:
                callback(percentage, calculate_features)
                previous_percentage = percentage

    # Compute optional percentage
    if callback is not None:
        callback(100, calculate_features)
    
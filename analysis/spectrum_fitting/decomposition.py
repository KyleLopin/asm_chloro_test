# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.utils import extmath


def pca(X, n_components=2, robust=False):
    # from https://nirpyresearch.com/pca-kernel-pca-explained/
    # and modified with https://nirpyresearch.com/robust-pca/
    # Preprocessing - Standard Scaler
    X_std = StandardScaler().fit_transform(X)

    # Calculate covariance matrix
    if robust:
        robust_cov = MinCovDet().fit(StandardScaler(with_std=False).fit_transform(X))
        cov_matrix = robust_cov.covariance_
    else:
        cov_matrix = np.cov(X_std.T)

    # Get eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # flip eigenvectors' sign to enforce deterministic output
    eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)

    # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
    matrix_w = np.column_stack([eig_vecs[:, -i] for i in range(1, n_components + 1)])

    # Get the PCA reduced data
    Xpca = X_std.dot(matrix_w)

    return Xpca


def robust_pca(X, n_components=2):
    # from https://nirpyresearch.com/pca-kernel-pca-explained/
    # and modified with https://nirpyresearch.com/robust-pca/
    # Preprocessing - Standard Scaler
    X_std = StandardScaler().fit_transform(X)

    # Calculate covariance matrix
    robust_cov = MinCovDet().fit(StandardScaler(with_std=False).fit_transform(X))
    cov_matrix = robust_cov.covariance_

    # Get eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # flip eigenvectors' sign to enforce deterministic output
    eig_vecs, _ = extmath.svd_flip(eig_vecs, np.empty_like(eig_vecs).T)

    # Concatenate the eigenvectors corresponding to the highest n_components eigenvalues
    matrix_w = np.column_stack([eig_vecs[:, -i] for i in range(1, n_components + 1)])

    # Get the PCA reduced data
    Xpca = X_std.dot(matrix_w)

    return Xpca


import numpy as np


def random_gaussian(dim):
    mu0 = np.zeros(dim)
    sigma0 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
    Q, _ = np.linalg.qr(sigma0)
    D = np.diag(2 * np.abs(np.random.normal(0, 1, dim)) + 0.01)
    sigma0 = np.dot(np.dot(Q, D), np.transpose(Q))
    return mu0, sigma0


def compute_roto_translation(gauss0, target_sKL):
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    dim = len(mu0)
    shift = np.zeros(dim)
    if dim == 1:
        rot = 1
        angles = 0
        num_angles = 1
        P = 1
    else:
        num_angles = np.int(np.floor(dim / 2))
        A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim)
        P, _ = np.linalg.qr(A)

        angles = -np.pi / 2 * np.random.rand(num_angles) + np.pi / 2
        Q = generate_canonical_rotation(angles, dim)
        rot = np.dot(np.dot(P, Q), np.transpose(P))

    gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
    sKL = symmetric_kullback_leibler(gauss0, gauss1)

    versor = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    versor = versor / np.sqrt(np.sum(versor ** 2))

    to_decrease = sKL > target_sKL

    while to_decrease:
        angles = angles / 2
        Q = generate_canonical_rotation(angles, dim)
        rot = np.dot(np.dot(P, Q), np.transpose(P))

        gauss1 = rotate_and_shift_gaussian(gauss0, rot, shift)
        sKL = symmetric_kullback_leibler(gauss0, gauss1)
        to_decrease = sKL > target_sKL

    sigma1 = gauss1[1]
    I = np.identity(dim)
    a = np.dot(versor, np.linalg.solve(sigma1, versor)) + np.dot(versor, np.linalg.solve(sigma0, versor))
    b = np.dot(np.dot(versor, np.linalg.solve(sigma1, rot - I)), mu0) + np.dot(np.dot(versor, np.linalg.solve(sigma0, rot - I)), mu0)
    c = 2 * sKL - 2 * target_sKL
    rho = (-b + np.sqrt(b ** 2 - a * c )) / a
    shift = rho * versor

    return rot, shift


def rotate_and_shift_gaussian(gauss, rot, shift):
    mu0 = gauss[0]
    sigma0 = gauss[1]
    mu1 = np.dot(mu0, np.transpose(rot)) + shift
    sigma1 = np.dot(np.dot(rot, sigma0), np.transpose(rot))
    sigma1 = 0.5 * (sigma1 + np.transpose(sigma1))
    return mu1, sigma1


def generate_canonical_rotation(angles, dim):
    Q = np.identity(dim)
    if dim == 1:
        return Q

    for i_angles in range(len(angles)):
        theta = angles[i_angles]
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        Q[2 * i_angles: 2 * (i_angles + 1), 2 * i_angles: 2 * (i_angles + 1)] = R

    return Q


def kullback_leibler(gauss0, gauss1):
    mu0 = gauss0[0]
    sigma0 = gauss0[1]
    mu1 = gauss1[0]
    sigma1 = gauss1[1]
    dim = len(mu0)
    return 0.5 * (np.trace(np.linalg.solve(sigma1, sigma0)) + np.dot(mu1 - mu0, np.linalg.solve(sigma1, mu1 - mu0)) - dim + np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0)))


def symmetric_kullback_leibler(gauss0, gauss1):
    return kullback_leibler(gauss0, gauss1) + kullback_leibler(gauss1, gauss0)


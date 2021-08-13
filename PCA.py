import numpy as np
import matplotlib.pyplot as plt


ROWS_DIM = 100
COLS_DIM = 100
SIGMAS = [0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]


def get_samples():
    A = np.random.randn(ROWS_DIM, COLS_DIM)
    A_mean_mat = np.tile(np.mean(A, 0).reshape((1, -1)), (ROWS_DIM, 1))
    A -= A_mean_mat
    U, _, V = np.linalg.svd(A)
    D = np.sqrt(2 * np.arange(10, 0, -1))
    X = np.dot(np.dot(U[:, :10], np.diag(D)), V[:10, :])
    return X


X = get_samples()


def get_noised_samples(sigma):
    Z = np.random.normal(0, sigma ** 2, (ROWS_DIM, COLS_DIM))
    return X + Z


def get_eigenvals(sigma):
    Y = get_noised_samples(sigma)
    return np.flip(np.linalg.eigvalsh(np.cov(Y)))


def plot_eigenvals(sigma):
    eigenvals = get_eigenvals(sigma)
    indices = np.arange(len(eigenvals)) + 1
    plt.bar(indices, eigenvals)
    plt.xlabel('Indices')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues of noisy data with standard deviation of {}'.
              format(sigma))
    plt.show()


def plot_all_eigenvals():
    for sigma in SIGMAS:
        plot_eigenvals(sigma)


# plot_all_eigenvals()


def determine_original_eigenvals_amount_helper(eigenvals):
    counter = 1
    for i in range(len(eigenvals) - 1):
        curr = eigenvals[i]
        next = eigenvals[i + 1]
        if 0.95 * curr > next and next > 0.01 * curr:
            counter += 1
        else:
            break
    return counter


def determine_original_eigenvals_amount():
    print()
    for sigma in SIGMAS:
        print('     Amount of original eigenvalues with '
              'standard devition of {} is '
              .format(sigma), determine_original_eigenvals_amount_helper(
            get_eigenvals(sigma)))


# determine_original_eigenvals_amount()


def get_original_and_noised_product_helper(sigma):
    Y = get_noised_samples(sigma)
    U, _, V = np.linalg.svd(X)
    X_left_singular_vecs = U[:10]
    Y_cov_eigenvecs = np.linalg.eig(np.cov(Y))[1][:10]
    inner_product = []
    for i in range(10):
        sing = X_left_singular_vecs[i]
        eig = Y_cov_eigenvecs[i]
        inner_product.append(sing @ eig)
    return inner_product


def get_original_and_noised_product():
    print()
    for sigma in SIGMAS:
        print()
        print('The inner product between the ten leading left singular '
              'vectors of X\nwith the ten leading eigenvectors of the '
              'covariance matrix of Y\nwith standard deviation of {} is '
              .format(sigma), np.round(get_original_and_noised_product_helper(
            sigma), 3))


# get_original_and_noised_product()


def run_all():
    plot_all_eigenvals()
    determine_original_eigenvals_amount()
    get_original_and_noised_product()


# run_all()

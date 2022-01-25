import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy.linalg as LA
from matplotlib import pyplot as plt


np.random.seed(0)

n_samples, n_features = 10000, 20
rng = np.random.RandomState(0)
X, y = make_regression(n_samples, n_features, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def x_update(rho, x_old, z_old, u_old, eta, iter):
    x_new = (rho * (z_old - u_old) + eta * x_old) / (np.dot(X_train[iter].T, X_train[iter]) + rho + eta)
    return x_new


def z_update(lamb, rho, x, u_old):
    z_new = np.zeros(len(x))

    k = x + u_old
    ratio = lamb / rho

    val = 0

    for i in range(len(x)):
        if k[i] > ratio:
            val = k[i] - ratio
        elif np.sum(x) <= ratio:
            val = 0
        elif k[i] < -ratio:
            val = k[i] + ratio

        z_new[i] = val

    return z_new


def u_update(u_old, x, z):
    u_new = u_old + x - z
    return u_new


def get_loss(x, z, lamb, iter):
    loss = np.square(LA.norm(np.dot(X_train[iter].T, x))) + lamb * np.sum(z)
    return loss


def get_const_violation(x, z):
    val = LA.norm(x - z)
    return val


def run_oadm(rho, eta, lamb, T, d):

    x = np.random.rand(d)
    z = np.random.rand(d)
    u = 0

    t = 0

    losses = []
    const_violations = []

    while t < T:
        x_new = x_update(rho, x, z, u, eta, t)
        z_new = z_update(lamb, rho, x_new, u)
        u_new = u_update(u, x, z)

        loss = get_loss(x_new, z_new, lamb, t + 1)
        const_violation = get_const_violation(x_new, z_new)

        losses.append(loss)
        const_violations.append(const_violation)

        x = x_new
        z = z_new
        u = u_new

        t += 1

    return losses, const_violations


def main():

    rho = 1
    eta = 1
    lamb = 1
    T = 120
    d = X.shape[1]

    losses, const_violations = run_oadm(rho, eta, lamb, T, d)

    start = 2

    iters = np.array([i for i in range(T)])[start:]

    plt.plot(iters, np.array(losses)[start:], label='loss')
    plt.plot(iters, np.array(const_violations)[start:], label='constraint violation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

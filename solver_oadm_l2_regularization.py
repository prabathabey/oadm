import cvxpy as cp
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy.linalg as LA
from matplotlib import pyplot as plt

np.random.seed(0)

n_samples, n_features = 10000, 20
rng = np.random.RandomState(0)
X, y = make_regression(n_samples, n_features, random_state=rng)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X


def x_update(rho, x_old, z_old, y_old, A, B, c, eta, iter):
    d = len(x_old)
    x_old = x_old.reshape(d, 1)
    z_old = z_old.reshape(d, 1)
    A = A.reshape(d, 1)
    B = B.reshape(d, 1)
    x = cp.Variable((d, 1))
    f = cp.square(cp.norm2(cp.sum(cp.multiply(X_train[iter].reshape(d, 1), x))))

    breg = 0.5 * cp.square(cp.norm2(x - x_old))
    cost = f + y_old * (cp.sum(cp.multiply(A, x)) + cp.sum(cp.multiply(B, z_old)) - c) + 0.5 * rho * cp.square(
        cp.norm2(cp.sum(cp.multiply(A, x)) + cp.sum(cp.multiply(B, z_old)) - c)) + eta * breg
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    return x.value


def z_update(rho, x, y_old, A, B, c, lamb):
    d = len(x)
    B = B.reshape(d, 1)
    z = cp.Variable((d, 1))
    # L2 regularization
    g = lamb * cp.square(cp.norm2(z))

    cost = g + y_old * (np.dot(A.T, x).item() + cp.sum(cp.multiply(B, z)) - c) + 0.5 * rho * cp.square(
        cp.norm2(A.T @ x + cp.sum(cp.multiply(B, z)) - c))
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    return z.value


def y_update(y_old, rho, x, z, A, B, c):
    y = y_old + rho * (np.dot(A.T, x) + np.dot(B.T, z) - c)
    return y


def get_loss(x, z, lamb, iter):
    f = np.square(LA.norm(np.dot(X_train[iter].T, x)))
    g = lamb * np.square(LA.norm(z))
    return f + g


def get_const_violation(A, B, c, x, z):
    val = LA.norm(np.dot(A.T, x) + np.dot(B.T, z) - c)
    return val


def run_oadm(rho, eta, T, d, lamb):
    A = np.ones(d)
    B = -np.ones(d)
    c = 0

    x = np.random.rand(d)
    z = np.random.rand(d)
    y = 0

    t = 0

    losses = []
    const_violations = []

    while t < T:
        x_new = x_update(rho, x, z, y, A, B, c, eta, t)
        z_new = z_update(rho, x_new, y, A, B, c, lamb)
        y_new = y_update(y, rho, x, z, A, B, c)

        loss = get_loss(x_new, z_new, lamb, ++t)
        const_violation = get_const_violation(A, B, c, x_new, z_new)

        losses.append(loss)
        const_violations.append(const_violation)

        x = x_new
        z = z_new
        y = y_new

        t += 1

    return losses, const_violations


def main():
    rho = 0.1
    eta = 1
    lamb = 1
    T = 100
    d = X.shape[1]

    losses, const_violations = run_oadm(rho, eta, T, d, lamb)

    start = 0
    iters = np.array([i for i in range(T)])[start:]

    plt.plot(iters, np.array(losses)[start:], label='loss')
    plt.plot(iters, np.array(const_violations)[start:], label='constraint violation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

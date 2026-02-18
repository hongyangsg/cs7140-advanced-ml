import numpy as np
from sklearn.linear_model import LinearRegression
import torch

def generate_LR_task_data(num_samples, p, beta, sigma):
    X = np.random.normal(0, 1, size = (num_samples, p))  # features in range [0, 10)
    y = X @ beta + np.random.normal(0, sigma, num_samples)
    return X, y


def generate_condition_matrix(p, r):
    U, _, Vt = np.linalg.svd(np.random.normal(0, 1, size = (p, p)))
    singular_values = np.zeros(p)
    singular_values[:r] = p / r  # set top r singular values to p/r, so that trace is p
    condition_mat = U @ np.diag(singular_values) @ Vt
    return condition_mat


def generate_LR_conditioned_task_data(num_samples, p, beta, sigma, condition_mat):
    X_raw = np.random.normal(0, 1, size = (num_samples, p))
    X = X_raw @ condition_mat # condition_mad is a p x p matrix
    y = X @ beta + np.random.normal(0, sigma, num_samples)
    return X, y


def hps_estimator(inputs, tests):
    X_source, y_source, X_target, y_target = inputs
    X_target_test, y_target_test = tests
    
    X = np.vstack([X_source, X_target])
    y = np.hstack([y_source, y_target])

    # Train
    model = LinearRegression()
    model.fit(X, y)

    test_mse = np.mean((model.predict(X_target_test) - y_target_test) ** 2)
    print(f"HPS test MSE on target test data: {test_mse}")
    return test_mse


def sps_estimator(inputs, tests, p):
    X_source, y_source, X_target, y_target = inputs
    # X_target_test, y_target_test = tests

    X_source = torch.tensor(X_source, dtype=torch.float32)    
    y_source = torch.tensor(y_source, dtype=torch.float32)
    X_target = torch.tensor(X_target, dtype=torch.float32)    
    y_target = torch.tensor(y_target, dtype=torch.float32)

    n1 = X_source.shape[0]
    n2 = X_target.shape[0]

    beta = torch.zeros(p, requires_grad = True)
    z = torch.zeros(p, requires_grad = True)

    eta = 0.01
    lam = 0.01
    print(X_source.shape)
    for i in range(100):
        loss = (torch.norm(X_source @ (beta + z) - y_source)**2 + torch.norm(X_target @ beta - y_target)**2) / (n1 + n2) + lam * torch.norm(z)**2
        loss.backward()

        #loss.step()

        with torch.no_grad():
            beta -= eta * beta.grad
            z -= eta * z.grad

        beta.grad.zero_()
        z.grad.zero_()


def ols_estimator(inputs, tests):
    X_target, y_target = inputs
    X_target_test, y_target_test = tests
    
    # Train only on target data
    model = LinearRegression()
    model.fit(X_target, y_target)

    test_mse = np.mean((model.predict(X_target_test) - y_target_test) ** 2)
    print(f"OLS test MSE on target test data: {test_mse}")
    return test_mse

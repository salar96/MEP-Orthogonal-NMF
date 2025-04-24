import numpy as np
import matplotlib.pyplot as plt
# from Deterministic_Annealing import DA
# import utils

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans

import pandas as pd


from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.stats.mstats import winsorize

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

import time
import cvxpy as cp
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp



def cluster_beta(X_init,Y_init,m,beta,pert):
    N, d = np.shape(X_init)
    Y_init += pert * (np.arange(m)+1).reshape(-1,1) * np.ones((m,d))
    X_expanded = np.expand_dims(X_init, 1)       # Shape (N, 1, d)
    Y_expanded = np.expand_dims(Y_init, 0)       # Shape (1, m, d)

    # Compute the squared differences, sum over the coordinate axis (axis=2)
    distances = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    distances_n = distances - np.min(distances, axis = 1).reshape(-1,1)
    P_yx = np.exp(-beta * distances_n)/(np.sum(np.exp(-beta * distances_n),axis = 1).reshape(-1,1))
    P_xy = P_yx/(np.sum(P_yx, axis = 0).reshape(1,-1))
    Y_c = P_xy.T @ X_init
    return Y_c, P_yx, P_xy

def clustering(X_init,m,pert,beta_final,alpha):
    beta = 0.001
    N, d = X_init.shape
    Y_init = np.mean(X_init,axis = 0)*np.ones((m,d))
    while beta <= beta_final:
        Y, P_yx, P_xy = cluster_beta(X_init,Y_init,m,beta,pert)
        beta *= alpha 
        Y_init = Y
        # print('beta: ', beta)
    return Y, P_yx

def Free_Energy(X, Y, q, P_ylx, beta):
    # Computation of the free energy
    N = X.shape[0]
    m = Y.shape[0]
    X_expanded = np.expand_dims(X, 1)       # Shape (N, 1, 2)
    Y_expanded = np.expand_dims(Y, 0)       # Shape (1, m, 2)
    distances = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    Px = np.diag(q)
    Py = np.sum(Px @ P_ylx, axis = 0)
    F = np.sum(Px @ np.multiply(P_ylx, distances + 1/beta * np.log(P_ylx)))

    return F

def F_Fdot_clustering(X, Y, q, P_ylx, u_y, u_p, u_b, beta):
    # Computation of the free energy
    N = X.shape[0]
    m = Y.shape[0]
    X_expanded = np.expand_dims(X, 1)       # Shape (N, 1, 2)
    Y_expanded = np.expand_dims(Y, 0)       # Shape (1, m, 2)
    distances = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    Px = np.diag(q)
    Py = np.sum(Px @ P_ylx, axis = 0)
    F = np.sum(Px @ np.multiply(P_ylx, distances + 1/beta * np.log(P_ylx)))
    F_shifted = F + 1/beta * np.log(m)

    # Time derivatives
    ## w.r.t Y: sum_{j=1}^m (dF/dy_j)^T u_j
    dFdY = 2 * (Py[:, np.newaxis] * Y - P_ylx.T @ Px @ X)
    dotF_y = cp.sum(cp.multiply(dFdY,u_y))

    ## w.r.t P_{y|x}: \sum_{i,j} dF/dp_{j|i} v_ij
    dotF_p = cp.sum(cp.multiply(Px @ (distances + 1/beta * np.log(P_ylx) + 1/beta), u_p))

    ## w.r.t beta: (1/beta)^2 H
    dFdb = -1/beta**2 * np.sum(Px @ np.multiply(P_ylx, np.log(P_ylx)))
    dotF_beta = dFdb * u_b

    # Total time derivative
    Fdot = dotF_y + dotF_p + dotF_beta

    return F_shifted, Fdot

def h_hdot(P_ylx, u_p):
    h = P_ylx * (1 - P_ylx)
    hdot = cp.multiply((1 - 2 * P_ylx), u_p)
    return h, hdot

def l_ldot(c, q, P_ylx, u_p):
    l = c - np.dot(q, P_ylx)
    ldot = - (q[:,np.newaxis].T @ u_p).flatten()
    return l, ldot

def control_dyn(X, Y, q, P_ylx, beta, u_b, p, gamma, alpha_h, alpha_l, c):
    
    N, d = X.shape
    m = Y.shape[0]
    # Decision variables 
    u_p = cp.Variable((N,m))
    u_y = cp.Variable((m,d))  
    # u_y = cp.Parameter((m,2))
    # u_y.value = np.zeros((m,2))
    delta = cp.Variable(1)
    
    ## Constant parameters of the QP
    # p = 1000
    # gamma = 1
    # alpha_h = 10
    # alpha_l = 10

    # Objective: minimize the sum of squares of x and the sum of q
    objective = cp.Minimize(cp.sum_squares(u_p) + cp.sum_squares(u_y) + p * delta**2)

    # Define constraints
    F , Fdot = F_Fdot_clustering(X, Y, q, P_ylx, u_y, u_p, u_b, beta)
    h, h_dot = h_hdot(P_ylx, u_p)
    l, l_dot = l_ldot(c, q, P_ylx, u_p)

    constraints = [
        Fdot <= -gamma * F + delta,
        cp.sum(u_p, axis=1) == 0,
        h_dot >= -alpha_h * h,
        l_dot >= -alpha_l * l
    ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solver Options for OSQP
    solver_options = {
        'max_iter': 50000,         # Increase max iterations to 20000
        'eps_abs': 1e-4,           # Adjust absolute tolerance
        'eps_rel': 1e-4,           # Adjust relative tolerance
        'eps_prim_inf': 1e-3,      # Adjust primal infeasibility tolerance
        'eps_dual_inf': 1e-3,      # Adjust dual infeasibility tolerance
        'verbose': False           # Enable verbose output to track solver progress
    }

    # Solve the problem using OSQP with customized options
    result = problem.solve(solver = 'OSQP', **solver_options)
    
    # Check the results
    if np.isnan(problem.value).any() == True:
        print("Nan encountered!")
        return np.zeros((N,m)), np.zeros((m,d)), F, 0
    else:         
        return u_p.value, u_y.value, F, Fdot.value
    
def project_to_stochastic_matrix(matrix):
    """
    Projects each row of the input matrix onto the simplex (triangle) defined by:
    - The row sums to 1
    - Each element in the row lies within [0, 1]

    :param matrix: np.ndarray, the input matrix (shape: NxM)
    :return: np.ndarray, the projected matrix (same shape as input)
    """
    def project_to_plane(vector):
        N = vector.shape[0]
        normal = np.ones((N,))
        p = 1 / N * np.ones((N,))
        n_dot_n = N
        v_dot_n = np.dot(vector, normal)
        p_dot_n = np.dot(p, normal)
        projection = vector - ((v_dot_n - p_dot_n) / n_dot_n) * normal
        return projection

    def project_to_triangle(v):
        tol = 1e-7
        v_proj = project_to_plane(v)
        v_clamped = np.clip(v_proj, tol, 1 - tol)
        sum_clamped = np.sum(v_clamped)

        if sum_clamped == 1:
            return v_clamped
        elif sum_clamped < 1:
            deficit = 1 - sum_clamped
            free_indices = v_clamped < 1
            num_free = np.sum(free_indices)
            if num_free > 0:
                increment = deficit / num_free
                v_clamped[free_indices] += increment
            return v_clamped
        else:
            return v_clamped / sum_clamped

    # Apply the projection to each row of the matrix
    projected_matrix = np.apply_along_axis(project_to_triangle, axis=1, arr=matrix)
    return projected_matrix

def dynamics_v2(z, X, beta, q, p, gamma, alpha_h, alpha_l, c, N, m):
    """
    Computes the time derivative of the state.
    
    Parameters:
      t       : time (not used explicitly here, but required by solve_ivp)
      z       : flattened state vector [Y.flatten(), P.flatten(), beta]
      X       : initial parameter or state used in control_dyn
      q, u_b, p, gamma, alpha_h, alpha_l : additional parameters for control_dyn
      N       : mumber of data poits
      m       : number of facilities
      
    Returns:
      dz/dt  : flattened derivative vector.
    """
    # Unpack the state vector
    d = X.shape[1]
    n_Y = m * d
    n_P = N * m
    Y = z[0:n_Y].reshape(m,d)
    P = z[n_Y:n_Y+n_P].reshape(N,m)
    u_b = 0
    

    # Compute control inputs; control_dyn should return u_p and u_y for state evolution.
    u_p, u_y, F_b, Fdot_b = control_dyn(X, Y, q, P, beta, u_b, p, gamma, alpha_h, alpha_l, c)
    
    # Compute the derivatives
    dY_dt = u_y           # dY/dt = u_y
    dP_dt = u_p           # dP/dt = u_p
    
    # Flatten the derivatives into a single vector
    dzdt = np.concatenate([dY_dt.flatten(), dP_dt.flatten()])
    return dzdt, F_b, Fdot_b

def adaptive_euler_forward(dynamics, z0, X, beta, q, p, gamma, alpha_h, alpha_l, c, N, m, T_f,
                           dt_init=0.01, dt_min=1e-4, dt_max=0.1, Ftol=1e-4, Ztol=1e-4, allowPrint=False):
    """
    Adaptive Euler integration with projection step for P based on Adaptive Gradient Descent.
    Includes stopping criteria based on both function value and state change.

    Parameters:
      dynamics : function returning dz/dt, F, Fdot
      z0       : initial state
      T_f      : final time
      dt_init  : initial step size
      dt_min   : minimum allowable step size
      dt_max   : maximum allowable step size
      Ftol     : stopping threshold on change in F
      Ztol     : stopping threshold on change in z
      allowPrint : if True, prints debug info

    Returns:
      z : final state after integration
    """

    import numpy as np

    np.random.seed(2)
    d = (len(z0) - N * m) // m
    z_prev = z0
    dt_prev = dt_init
    theta_prev = np.inf
    t = 0.0
    iter_count = 0
    pert = 1e-10

    F_prev = np.inf
    z_old = np.copy(z_prev)  # Initialize z_old to something

    while t < T_f:
        dzdt, F, Fdot = dynamics(z_prev, X, beta, q, p, gamma, alpha_h, alpha_l, c, N, m)

        # === New termination criteria based on F and z ===
        dz = np.linalg.norm(z_prev - z_old) if iter_count > 0 else np.inf

        if abs(F - F_prev) < Ftol and dz < Ztol:
            if allowPrint:
                print(f"Ftol and Ztol successful\titer:{iter_count}\ttime {t:.3e}\t"
                      f"Fdiff={abs(F - F_prev):.3e} < Ftol={Ftol:.3e}\t"
                      f"dz={dz:.3e} < Ztol={Ztol:.3e}")
            break
        elif allowPrint:
            print(f"t:{t:.3e}\tF:{F:.6f}\tFdiff:{abs(F - F_prev):.6f}\tdt:{dt_prev:.5f}\t"
                  f"dzdt_norm:{np.max(np.abs(dzdt)):.6f}\tdz:{dz:.3e}")

        # Adaptive step size
        if iter_count > 0:
            step_size_1 = np.sqrt(1 + theta_prev) * dt_prev
            grad_diff = np.linalg.norm(dzdt - dzdt_old) + 1e-6  # prevent div by zero
            step_size_2 = np.linalg.norm(z_prev - z_old) / (2 * grad_diff)
            dt = min(max(step_size_1, dt_min), dt_max)
        else:
            dt = dt_init

        # Euler update
        z_next = z_prev + dt * dzdt

        # Projection step for P
        n_Y = m * d
        n_P = N * m
        Y_next = z_next[:n_Y].reshape(m, d) + pert * np.random.rand(m, d)
        P_next = z_next[n_Y:].reshape(N, m)

        # Apply projection to P
        P_projected = project_to_stochastic_matrix(P_next)

        # Reassemble projected state
        z_projected = np.concatenate([Y_next.flatten(), P_projected.flatten()])

        # Theta update
        theta = dt / dt_prev if iter_count > 0 else 1.0

        # Prepare for next iteration
        z_old = z_prev
        dzdt_old = dzdt
        dt_prev = dt
        theta_prev = theta
        z_prev = z_projected
        F_prev = F
        t += dt
        iter_count += 1

    return z_prev


def closest_binary_matrix(P):
    """
    Converts a row-stochastic matrix P into its closest binary matrix.
    
    Parameters:
    P (numpy.ndarray): A (N, K) row-stochastic matrix where rows sum to 1.
    
    Returns:
    numpy.ndarray: A (N, K) binary matrix with one 1 per row.
    """
    binary_matrix = np.zeros_like(P)  # Initialize a binary matrix with zeros
    max_indices = np.argmax(P, axis=1)  # Get the index of the max element in each row
    binary_matrix[np.arange(P.shape[0]), max_indices] = 1  # Set the max index to 1
    
    return binary_matrix

def get_permutation_matrix(A):
    """
    Returns the permutation matrix that sorts the columns of A in descending order of column sums.

    Parameters:
    A (numpy.ndarray): The matrix whose column sums determine the permutation.

    Returns:
    P (numpy.ndarray): The permutation matrix.
    """
    # Compute column sums
    col_sums = np.sum(A, axis=0)

    # Get column order indices (sorted in descending order)
    sorted_indices = np.argsort(-col_sums)  # Negative sign for descending order

    # Create the permutation matrix
    P = np.eye(A.shape[1])[:, sorted_indices]

    return P

def solve_constrained_clustering(Xa_n, m, C, pert=1e-6, beta_init=0.001, beta_final=50, beta_factor=1.5):
    """
    Solves the constrained clustering problem for a given normalized dataset Xa_n.

    Parameters:
    - Xa_n (numpy.ndarray): Normalized data matrix of shape (N, d).
    - m (int): Number of clusters.
    - pert (float): Perturbation level for initialization.
    - beta_final (float): Final beta value.
    - alpha (float): Clustering parameter.

    Returns:
    - Ya_n_c (numpy.ndarray): Cluster centers (in normalized space).
    - Pa_n_c_bin (numpy.ndarray): Binary assignment matrix.
    - Pi (numpy.ndarray): Permutation matrix.
    - D_constrained (float): Clustering cost.
    - H_constrained (numpy.ndarray): H matrix.
    - Py_constrained (numpy.ndarray): Assignment vector.
    """

    N, d = Xa_n.shape  # Extract dimensions

    # Step 1: Initialize Y and P
    Y0 = np.mean(Xa_n, axis=0) * np.ones((m, d)) + pert * np.random.rand(m, d)
    P0 = np.random.rand(N, m)
    P0 = P0 / P0.sum(axis=1, keepdims=True)

    q = 1 / N * np.ones(N)
    Px = np.diag(q)
    Py = np.sum(Px @ P0, axis=0)
    z0_init = np.concatenate([Y0.flatten(), P0.flatten()])
    z0 = np.copy(z0_init)  # Ensure z0 is not modified in-place

    # Step 2: Set up parameters for constrained clustering
    alpha_h = 40
    alpha_l = 20
    p = 10
    gamma = 1

    # Estimate maximum iterations
    max_iterations = int(np.ceil(np.log(beta_final / beta_init) / np.log(beta_factor))) + 1

    # Preallocate storage
    z_values = np.zeros((max_iterations, z0.shape[0]))
    beta_values = np.zeros(max_iterations)

    # Store initial values
    z_values[0, :] = z0_init
    beta_values[0] = beta_init

    # Iteratively solve for constrained clustering
    beta = beta_init
    counter = 1
    c = C * np.ones(m)

    while beta < beta_final:
        print(f"Iteration {counter}: Solving for beta = {beta}")

        # Solve constrained optimization using Euler forward integration
        z = adaptive_euler_forward(
            dynamics_v2, z0, Xa_n, beta, q, p, gamma, alpha_h, alpha_l, c, N, m,
            T_f=15, dt_init=0.01, dt_min=1e-4, dt_max=0.1,
            Ftol=1e-4, Ztol=1e-4, allowPrint=False
        )

        
        # Store results
        beta_values[counter] = beta
        z_values[counter, :] = z  # Store z in preallocated array

        # Increase beta geometrically
        beta *= beta_factor

        # Update initial z for the next iteration
        z0 = np.copy(z)

        counter += 1

    # Trim storage arrays
    beta_values = beta_values[:counter]
    z_values = z_values[:counter, :]

    # Extract final clustering results
    n_Y = m * d
    n_P = N * m
    Ya_n_c_2 = z_values[-1, :n_Y].reshape(m, -1)  # Cluster centers
    Pa_n_c = z_values[-1, n_Y:].reshape(N, m)  # Assignment matrix

    # Convert assignment matrix to binary
    Pa_n_c_bin = closest_binary_matrix(Pa_n_c)

    # Compute permutation matrix
    Pi = get_permutation_matrix(Pa_n_c_bin)

    return Pi.T @ Ya_n_c_2, Pa_n_c_bin @ Pi

def Clustering_Cost(X, Y, P, Px=None):
    N = X.shape[0]
    
    # Expand dimensions for broadcasting
    X_expanded = np.expand_dims(X, 1)  # Shape (N, 1, d)
    Y_expanded = np.expand_dims(Y, 0)  # Shape (1, m, d)
    
    # Compute squared Euclidean distances
    distances = np.sum((X_expanded - Y_expanded) ** 2, axis=2)
    
    # Set Px only if it is not specified
    if Px is None:
        Px = (1 / N) * np.eye(N)
    
    # Compute clustering cost
    D = np.sum(Px @ np.multiply(P, distances))
    
    return D


def update_P(P, X, W):
    N, m = P.shape  # Dimensions
    _, d = X.shape  # Feature dimension

    PW = P @ W  # Shape: (N, d)
    dot_products = np.einsum('ij,ij->i', PW, X)  # Shape: (N,)
    PW_norm_sq = np.sum(PW ** 2, axis=1)  # Shape: (N,)

    # Compute V values
    V = dot_products / (PW_norm_sq + 1e-10)  # Shape: (N,)

    nonzero_indices = np.argmax(P, axis=1)  # Shape: (N,)
    P_updated = np.copy(P)

    # Replace only the nonzero elements
    P_updated[np.arange(N), nonzero_indices] = V

    return P_updated
import numpy as np
import matplotlib.pyplot as plt
import math

# Saturation function for control input smoothing
def sat_epsilon(x, epsilon=0.1):  # default epsilon is 0.1, but you can adjust as needed
    if x > epsilon:
        return 1
    elif x < -epsilon:
        return -1
    else:
        return x / epsilon
sat_epsilon_vec = np.vectorize(sat_epsilon)

# System dynamics function
def dynamics(state, tau_rot, tau_trans, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R):
    q = state[:4]
    omega = state[4:7]
    r = state[7:10]
    v = state[10:13]

    v_r = v - R @ v_c  # Relative velocity

    # Rotational dynamics
    M_drag = -drag_linear @ omega - drag_quadratic @ (np.abs(omega * omega))
    omega_dot = J_inv @ (tau_rot + M_drag - np.cross(omega, J @ omega))
    q_dot = 0.5 * quaternion_multiply(q, [0, *omega])

    # Translational dynamics
    r_dot = v
    F_drag = -drag_linear_trans @ v_r - drag_quadratic_trans @ (np.abs(v_r) * v_r)
    F_buoyancy = rho * V * g
    F_external = tau_trans - F_drag + F_buoyancy - m * g
    v_dot = np.linalg.inv(M_total) @ F_external - np.cross(omega, v_r)

    return np.hstack([q_dot, omega_dot, r_dot, v_dot])

# Sliding Mode Control (SMC) function
def smc_controller_full(q, omega, r, v, r_desired, v_desired, lambda_rot, lambda_trans, K_rot, B_rot, K_trans, B_trans, v_c, R):

    v_r = v - R @ v_c  # Relative velocity
    M_drag = -drag_linear @ omega - drag_quadratic @ (np.abs(omega * omega))
    F_drag = -drag_linear_trans @ v_r - drag_quadratic_trans @ (np.abs(v_r) * v_r)

    q_desired = np.array([0, 0, 0, 1])
    q_conjugate = np.array([q[0], -q[1], -q[2], -q[3]])
    error_q = quaternion_multiply(q_desired, q_conjugate)
    error_omega = omega - np.zeros(3)
    s_rot = error_omega + lambda_rot * error_q[:3]  # vector part of quaternion error
    tau = -K_rot @ s_rot - B_rot @ sat_epsilon_vec(s_rot) - M_drag
    error_r = r - r_desired
    error_v = v - v_desired
    s_trans = error_v + lambda_trans * error_r
    f_thrust = -K_trans @ s_trans - B_trans @ sat_epsilon_vec(s_trans) + M_total @ F_drag + M_total @ np.cross(omega, v_r)

    return tau, f_thrust

# RK4 Integration for updating the dynamics
def rk4_step(func, state, tau_rot, tau_trans, dt, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R):
    k1 = dt * func(state, tau_rot, tau_trans, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R)
    k2 = dt * func(state + 0.5 * k1, tau_rot, tau_trans, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R)
    k3 = dt * func(state + 0.5 * k2, tau_rot, tau_trans, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R)
    k4 = dt * func(state + k3, tau_rot, tau_trans, J_inv, J, drag_linear, drag_quadratic, drag_linear_trans, drag_quadratic_trans, v_c, R)

    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Quaternion multiplication
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
    ])
    return R

# Constants for the simulation
rho = 1025  # Seawater density
V = 0.01 / 4  # AUV volume
g = np.array([0, 0, 9.81])
m = rho * V

# Added mass coefficients and matrix
m_axx = 0.36  # Adjust values based on system requirements
m_ayy = 1
m_azz = 1.5
M_added = np.diag([m_axx, m_ayy, m_azz])
M_total = m * (np.eye(3) + M_added)

# Drag coefficients for linear and quadratic damping models
X_u = 0.048
Y_v = 0
Z_w = 0.044
X_uu = 5.85
Y_vv = 11.98
Z_ww = 21.85
K_p = 0
M_q = 0
N_r = 21.85
K_pp = 5.85
M_qq = 11.98
N_rr = 21.85

# Linear and quadratic drag matrices
drag_linear_trans = np.diag([X_u, Y_v, Z_w])
drag_quadratic_trans = np.diag([X_uu, Y_vv, Z_ww])
drag_linear = np.diag([K_p, M_q, N_r])
drag_quadratic = np.diag([K_pp, M_qq, N_rr])

# Inertia matrix and its inverse
J_added = np.diag([0.0049 * 0.36, 0.023, 0.0021 * 1.15])
J_bar = np.diag([0.0049, 0.023, 0.0021])
J = np.eye(3) + J_added
J_inv = np.linalg.inv(J)

# Controller gains and parameters
lambda_gain = np.array([1, 1, 1])
K = np.diag([1, 1, 1])
B = np.diag([1, 1, 1])
K_trans = np.diag([0.2, 0.2, 0.2])
B_trans = np.diag([3.2, 3.2, 3.2])
lambda_trans = np.array([1, 1, 1])

# Initial states for leader and follower
state_leader = np.array([0, 0, 0, 1, 0.1, 2, 3, 0, 0, 1.1, 1, 0.1, 0.1])
state_follower = np.array([0, 0, 0, 1, 1, 2, 3, -1, 0, 1.1, 0.1, 0.1, 0.1])

# Desired positions and velocities
r_desired_leader = np.array([1.0, 2.0, 3.0])
v_desired_leader = np.zeros(3)
r_relative_desired = np.array([3.0, -1.0, 1.0])
v_relative_desired = np.zeros(3)

#

"""
Pointing constraint for the safety filter.

Author: Orfeas Koulamas - 2026
Modified by: Cemal Yilmaz - 2026
"""

import numpy as np
from scipy.optimize import fsolve
from agent_training.constants import get_constants

constants = get_constants()

absSq = lambda x: x * np.abs(x) # 2023 paper's "ssq" function used in eq. (13) & (28)

def R(q): # quaternion rotation matrix
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)    ],
                     [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)    ],
                     [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]], np.float32)

def skew(v): # 3x3 skew-symmetric matrix, skew(v) * x = v cross x, used for phi
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,    -v[0]],
                     [-v[1], v[0],  0   ]], np.float32)

def k(state, M_F):    # k = q^T M_F q, M_F is from Eq. (8) in "Engineering Notes" paper
    q = state[0:4].astype(np.float32)
    k = q.T @ M_F @ q      
    return k.astype(np.float32)

def kdot(state, M_F, omega_cross): # kdot = q^T M_F Ω(ω) q
    q = state[0:4].astype(np.float32)
    kdot = q.T @ M_F @ omega_cross @ q
    return kdot.astype(np.float32)

def phi(state, u, r_F, constants, n_F): # r_F: boresight vector in body frame
    q     = state[0:4]     # attitude quaternion
    omega = state[4:7]     # angular velocity
    w     = state[7:10]    # wheel velocities 
    wheel_positions = constants['A'] # to avoid confusion with constraint output A. 3x3 matrix
    Z = constants['Z']
    Jtot = constants['J_tot']
    J_w = constants['J_w'] # 3x3 matrix

    s = R(q).T @ n_F      #n_F: inertial sun direction (constant)

    # dynamics eq (2b) to get omegadot for given u and state
    omegadot = Z[0:3, :] @ np.concatenate([-np.cross(omega, Jtot @ omega + wheel_positions @ J_w @ w) , u])

    # same as matlab code for their sdot = sdotdot = 0
    phi = s @ (skew(omega) @ (skew(omega) @ r_F)) - s @ (skew(r_F) @ omegadot) 
    return phi.astype(np.float32)


def constraintQ(state, constants, index, outdata, n_F, theta_F, episode_count, episode_step):
    """
    Pointing constraint function.

    Args:
        state: current state
        constants: dictionary of constants needed for calculations
        index: index for storing results in outdata
        outdata: dictionary to store intermediate results for logging and debugging
        n_F: inertial KOZ direction (constant)
        theta_F: angle between boresight and KOZ direction
        episode_count: current episode count for logging
        episode_step: current episode step for logging
    Returns:
        res: A tuple containing (A, b, q_log) where:
        A: Linear term vector for control input in pointing constraint.
        b: Constant term for pointing constraint.
        q_log: Log string for any warnings or issues encountered during constraint calculation.
    """
    # Returns the A and b constraint matrices for ensuring avoidance is achieved.
    # Results should be fed into QP as  A*u ≤ b (QP probably Quadratic Programming (argmin part?) in CalculateU.m)

    r_F = constants['r_F']
    q_log = ""

    # M_F matrix components
    m1 = np.dot(r_F, n_F) - np.cos(theta_F) # scalar
    m2 = np.cross(r_F, n_F) #vector
    A_mf = (  np.outer(r_F, n_F) + np.outer(n_F, r_F) - (np.dot(r_F, n_F) + np.cos(theta_F))*np.eye(3)  ) # 3x3 matrix

    M_F = np.block([[m1, m2.reshape(1, -1)], # Build M_F matrix
                [m2.reshape(-1, 1), A_mf]]).astype(np.float32)

    omega = state[4:7].astype(np.float32)   # gets current values from state
    omega_cross = np.array([[0,        -omega[0], -omega[1], -omega[2]], # calculates Ω(ω) with current values
                            [omega[0],  0,         omega[2], -omega[1]], # used for kdot
                            [omega[1], -omega[2],  0,         omega[0]],
                            [omega[2],  omega[1], -omega[0],  0       ]], np.float32) 

    dt      = constants['dt']
    M2plus  = constants['M2plus']
    M3plus  = constants['M3plus']
    mu      = constants['mu']
    delta_2 = constants['delta_2']
    Delta_2 = constants['Delta_2']

    # Store k and kdot
    outdata['k'][index] = k(state, M_F)
    outdata['kdot'][index] = kdot(state, M_F, omega_cross)

    ### eq. (13)
    h = outdata['k'][index] + absSq(outdata['kdot'][index]) / (2 *mu)
    
    if h > 0: #???? figure out difference between h and H at Constraint.m  
        print('Excessive H',end=",")
        q_log += f"Excessive H at episode {episode_count}, step {episode_step}\n" 

    if outdata['k'][index] > 0:
        print('Excessive h',end=",")
        q_log += f"Excessive h at episode {episode_count}, step {episode_step}\n"  
    outdata['H'][index] = h


    ### eq. (27)
    Pk = lambda phi_val: (outdata['k'][index] + outdata['kdot'][index]*dt + 0.5*phi_val*dt**2 + 0.5*M2plus*dt**2 + 1/6*M3plus*dt**3)

    ### finds root P_k + δ2 = 0 (probably δ2 since later I have Delta2(written with capital D), which has to be Δ2)
    ## Should the starting guess for the root be 0?
    try:
        phi_req1 = fsolve(lambda y: Pk(y) + delta_2, 0)[0]
    except Exception as e:
        print(f"fsolve failed for phi_req1: {e}")
        q_log += f"fsolve failed for phi_req1: {e} at episode {episode_count}, step {episode_step}\n"
        phi_req1 = 0 # Default or error value

    ### phi call with u_zero is the ZERO control input (no torques).
    ### so b1 = phi_req1 - phi_zero = "how much MORE we need beyond zero control"(?)
    u_zero = np.array([0, 0, 0])
    b1 = phi_req1 - phi(state, u_zero, r_F, constants, n_F)


    ### eq. (28)
    Ph = lambda phi_val: Pk(phi_val) + (1/(2*mu)) * absSq( outdata['kdot'][index] + phi_val*dt + M2plus*dt + 0.5*M3plus*dt**2 )

    ### finds root Ph + Δ2 = 0, again: starting guess for the root is 0?
    try:
        phi_req2 = fsolve(lambda y: Ph(y) + Delta_2, 0)[0]
    except Exception as e:
        print(f"fsolve failed for phi_req2: {e}")
        q_log += f"fsolve failed for phi_req2: {e} at episode {episode_count}, step {episode_step}\n"
        phi_req2 = 0 # Default or error value

    ### "how much MORE we need beyond zero control"(?)
    b2 = phi_req2 - phi(state, u_zero, r_F, constants, n_F)

    ### takes the min. of the b1 and b2 - whichever is more restrictive (smaller value satisfies both constraints)(?)
    b = np.min([b1, b2])


    # Calculate A using a numerical gradient for phi, because it's linear rather than writing another function to compute it.
    A = np.zeros(3)
    for i in range(3):
        dist = 1.0 # arbitrary since it is linear
        e = np.zeros(3)
        f = np.copy(e) # Use np.copy to prevent modification of e
        f[i] = dist
        # Note: The original phi function must accept a 4-element array/list for the control input u
        A[i] = (phi(state, f, r_F, constants, n_F) - phi(state, e, r_F, constants, n_F)) / dist

    # A should be returned as a 1D array of 4 elements or converted to a 2D array (1x4) depending on the expected 
    # format for the QP solver. Will return it as a 1D numpy array for simplicity, as it's the result of the loop.
    return A, b, q_log
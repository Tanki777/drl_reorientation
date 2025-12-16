import numpy as np
from scipy.optimize import fsolve
from constants import get_constants

absSq = lambda x: x * np.abs(x) # paper's "ssq" function used in eq. (13) & (28)

def k(t, state, M_F):    # k = q^T M_F q, M_F is from Eq. (8) in "Engineering Notes" paper
    q = state[0:4].astype(np.float32)
    k = q.T @ M_F @ q      
    return k.astype(np.float32)

def kdot(t, state, M_F, omega_cross): # kdot = q^T M_F Ω(ω) q
    q = state[0:4].astype(np.float32)
    kdot = q.T @ M_F @ omega_cross @ q
    return kdot.astype(np.float32)

def kdotdot(t, state, M_F, omega_cross, omega_dot_cross): # kdotdot = 1/2 q^T Ω^T(ω) M_F Ω(ω) q  +  q^T M_F Ω(ωdot) q  +  1/2 q^T M_F Ω^2(ω) q 
    q = state[0:4].astype(np.float32)
    kdotdot = 1/2*(q.T @ omega_cross.T @ M_F @ omega_cross @ q)+(q.T @ M_F @ omega_dot_cross @ q) + 1/2*(q.T @ M_F @ omega_cross @ omega_cross @ q)
    return kdotdot.astype(np.float32) # possible optimisation of 3rd term: Ω^2(ω)=(-|ω|^2)I_4=-(ω1^2+ω2^2+ω3^2)I_4, where I_4 = 4x4 identity matrix    
# For no disturbances (xi=0), kdotdot is the 2023 paper code's 'phi' function. 
# Otherwise phi is the component of kdotdot that is certain (eq.(19) 2023 paper), and the implementation is different


def constraintQ(t, state, constants, index, outdata, omega_dot):
# Returns the A and b constraint matrices for ensuring avoidance is achieved.
# Results should be fed into QP as  A*u ≤ b (QP probably Quadratic Programming (argmin part?) in CalculateU.m)

    r_F = constants['r_F']
#!!!!!!!!!!!!!!!! TODO load n_F and theta_F that have been initialised for this episode !!!!!!!!!!!!!!!!

# M_F matrix components
    m1 = np.dot(r_F, n_F) - np.cos(theta_F) # scalar
    m2 = np.cross(r_F, n_F) #vector
    A_mf = (  np.outer(r_F, n_F) + np.outer(n_F, r_F) - (np.dot(r_F, n_F) + np.cos(theta_F))*np.eye(3)  ) # 3x3 matrix

    M_F = np.block([[m1, m2.reshape(1, -1)], # Build M_F matrix
                [m2.reshape(-1, 1), A_mf]]).astype(np.float32)

# Ω(ω) matrix from eq. (2a) 2023 paper, used for kdot and kdotdot
    def omega_cross_matrix(omega):
        return np.array([[0,        -omega[0], -omega[1], -omega[2]],
                         [omega[0],  0,         omega[2], -omega[1]],
                         [omega[1], -omega[2],  0,         omega[0]],
                         [omega[2],  omega[1], -omega[0],  0       ]], np.float32)

    omega = state[4:7].astype(np.float32)   # gets current values from state
    omega_cross = omega_cross_matrix(omega) # calculates Ω(ω) with current values
    omega_dot_cross = omega_cross_matrix(omega_dot) ## calculates Ω(ωdot). omega_dot was passed as arguement of constraintQ

    dt      = constants['dt']
    M2plus  = constants['M2plus']
    M3plus  = constants['M3plus']
    mu      = constants['mu']
    delta_2 = constants['delta_2']
    Delta_2 = constants['Delta_2']

# Store k and kdot
    outdata['k'][index] = k(t, state, M_F)                      # Assuming outdata['k'] is an array/list
    outdata['kdot'][index] = kdot(t, state, M_F, omega_cross)   # same

### eq. (13)
    h = outdata['k'][index] + absSq(outdata['kdot'][index]) / (2 *mu)
    
    if h > 0:
        print('Excessive H')
# Assuming k is a dictionary/array accessible within outdata Note: MATLAB's 'outdata.h' might be an array, 
# which means the comparison should be done carefully if it's meant to check the whole array. 
# translated to python as a check on the *current* k value at the given index, unless 'outdata.h' was meant to be the full array.

    if outdata['k'][index] > 0:
        print('Excessive h') #???? figure out difference between h and H at Constraint.m        
    outdata['H'][index] = h # Store the calculated ????



### eq. (27)
    Pk = lambda phi_val: (outdata['k'][index] + outdata['kdot'][index]*dt + 0.5*phi_val*dt**2 + 0.5*M2plus*dt**2 + 1/6*M3plus*dt**3)

### finds root P_k + δ2 = 0 (probably δ2 since later I have Delta2(written with capital D), which has to be Δ2)
## Gemini says "The starting guess for the root is 0"
    try:
        phi_req1 = fsolve(lambda y: Pk(y) + delta_2, 0)[0]
    except Exception as e:
        print(f"fsolve failed for phi_req1: {e}")
        phi_req1 = 0 # Default or error value

### phi(t,state,[0;0;0;0],p) IS THE ZERO CONTROL INPUT(no torques).
### so b1 = phi_req1 - phi_zero = "how much MORE we need beyond zero control"(?)
    u_zero = np.array([0, 0, 0, 0])
    b1 = phi_req1 - phi(t, state, u_zero, r_F)



### eq. (28)
    Ph = lambda phi_val: Pk(phi_val) + (1/(2*mu)) * absSq( outdata['kdot'][index] + phi_val*dt + M2plus*dt + 0.5*M3plus*dt**2 )

### finds root Ph + Δ2 = 0, #gemini says "The starting guess for the root is 0"
    try:
        phi_req2 = fsolve(lambda y: Ph(y) + Delta_2, 0)[0]
    except Exception as e:
        print(f"fsolve failed for phi_req2: {e}")
        phi_req2 = 0 # Default or error value

### "how much MORE we need beyond zero control"(?)
    b2 = phi_req2 - phi(t, state, u_zero, r_F)


### takes the min. of the b1 and b2 - whichever is more restrictive (smaller value satisfies both constraints)(?)
    b = np.min([b1, b2])



# Calculate A Using a numerical gradient for phi because it's linear rather than writing another function to compute it.
    A = np.zeros(4)
    for i in range(4):
        dist = 1.0 # arbitrary since it is linear
        e = np.zeros(4)
        f = np.copy(e) # Use np.copy to prevent modification of e
        f[i] = dist
    # Note: The original phi function must accept a 4-element array/list for the control input u
        A[i] = (phi(t, state, f, r_F) - phi(t, state, e, r_F)) / dist

# A should be returned as a 1D array of 4 elements or converted to a 2D array (1x4) depending on the expected 
# format for the QP solver. Will return it as a 1D numpy array for simplicity, as it's the result of the loop.
    return A, b

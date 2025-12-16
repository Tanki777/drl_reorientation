import numpy as np

def constraintE(t, state, constants, index, outdata): 
#Returns the H, A, b constraint matrices to be fed into QCQP as: u^T @ H @ u + A @ u + b ≤ 0
#Energy constraint function: η_ω = ω^T P ω - e_max ≤ 0, but 2023 paper's code used Je instead of P matrix

    Z = constants['Z']
    Je = constants['Je']
    dt = constants['dt']
    e_max = constants['e_max']
    M2omega = constants['M2omega']
    M1omega = constants['M1omega']
    
    
# Extract angular velocities from state !!! VERIFY indices are correct !!!
    omega = state[4:7]

# Energy constraint
    h_omega = omega.T @ Je @ omega - e_max

# Store current constraint value
    outdata['h'][index] = h_omega

    if outdata['h'][index] > 0:
        print('Excessive kinetic energy')
    
# Extract wheel-to-torque mapping (rows 0-2, cols 3-6)
    Z_sub = Z[0:3, 3:7]  # Shape: (3, 4)
    
    
    H = Z_sub.T @ Je @ Z_sub * dt**2  # Shape: (4, 4)
    
    A = 2 * omega.T @ Je @ Z_sub * dt  # Shape: (4,)
    A = A.flatten() #ensures A is 1D array
    
    b = - h_omega - M1omega*dt - 0.5*M2omega*dt**2
    
    return H, A, b

"""
Energy constraint for the safety filter.
Currently not used due to issues when used by the filter.

Author: Orfeas Koulamas - 2026
Modified by: Cemal Yilmaz - 2026
"""

def constraintE(state, constants, index, outdata):
    """
    Energy constraint function.

    Args:
        state: current state
        constants: dictionary of constants needed for calculations
        index: index for storing results in outdata
        outdata: dictionary to store intermediate results for logging and debugging

    Returns:
        res: A tuple containing (H, A, b) where:
        H: Quadratic term matrix for control input in energy constraint.
        A: Linear term vector for control input in energy constraint.
        b: Constant term for energy constraint.
    """
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

    # Warning
    if outdata['h'][index] > 0:
        #print('Excessive kinetic energy')
        pass
    
    # Extract wheel-to-torque mapping (rows 0-2, cols 3-6)
    Z_sub = Z[0:3, 3:6]  # Shape: (3, 3)
    
    
    H = Z_sub.T @ Je @ Z_sub * dt**2  # Shape: (3, 3)
    
    A = 2 * omega.T @ Je @ Z_sub * dt  # Shape: (3,)
    A = A.flatten() #ensures A is 1D array
    
    b = - h_omega - M1omega*dt - 0.5*M2omega*dt**2
    
    return H, A, b
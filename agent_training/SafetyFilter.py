import numpy as np
from scipy.optimize import minimize
import time
from constraintQ import constraintQ
from constraintE import constraintE
from constants import get_constants


outdata = {} # Define outdata globally first
initialized = False # Flag to track status

def initialize():
    global constants, outdata, initialized
    if initialized: # stop if already loaded
        return

    constants = get_constants()
    # Initialize dictionary keys
    outdata['k'] = [None] * 2   # Index 0: energy, Index 1: pointing
    outdata['kdot'] = [None] * 2
    outdata['H'] = [None] * 2
    outdata['h'] = [None] * 2
    
    initialized = True # Set flag to True


def safety_filter(wheels_desired, t, state): # gets control input, returns safe one

    initialize()
    global constants, outdata

    start = time.time()
    
# Compute constraints
    H0, A0, b0 = constraintE(t, state, constants, 0, outdata)  # Energy constraint
    A1, b1 = constraintQ(t, state, constants, 1, outdata)  # Pointing constraint
    
# Setup QP optimization
    nonlcon_omega = lambda u: u.T @ H0 @ u + A0 @ u - b0  # Nonlinear energy constraint
    
    scale = 1e5  # Scaling for numerical stability
    lower = -constants['u_max'] * np.ones(4) * scale
    upper = constants['u_max'] * np.ones(4) * scale
    
# Objective: minimize ||u - wheels_desired||^2
    F = -2 * wheels_desired
    
# Linear constraints: A*u <= b
    A = A1[np.newaxis, :]  # Shape (1, 4)
    b = np.array([b1])
    
    try:
    # Quadratic objective: u^T*u - 2*wheels_desired^T*u
        objective = lambda u: (u.T @ u / scale**2 + F.T @ u / scale) * 1e10
        
    # Linear inequality constraint: A*u <= b
        linear_constraint = {'type': 'ineq', 'fun': lambda u: b*scale - A @ u}
        
    # Nonlinear inequality constraint: nonlcon_omega(u) <= 0
        nonlin_constraint = {'type': 'ineq', 
                           'fun': lambda u: -(nonlcon_omega(u/scale) * 1e6)}
        
        result = minimize(objective, np.zeros(4), method='SLSQP',
                        bounds=[(lower[i], upper[i]) for i in range(4)],
                        constraints=[linear_constraint, nonlin_constraint],
                        options={'disp': False, 'ftol': 1e-8})
        
        u_safe = result.x / scale #VERIFY
        status = 1 if result.success else 0
        
    except Exception as e:
        print(f"Safety filter optimization error: {e}")
        u_safe = wheels_desired
        status = 0
    
    if status < 1:
        print(f'Safety filter optimization failed. Using desired control.')
    
    if np.any(np.isnan(u_safe)):
        print('Warning: NaN values in safe control output')
        u_safe = wheels_desired
    
# Store output data
    outdata['u'] = u_safe
    outdata['compute'] = time.time() - start
    
    return u_safe

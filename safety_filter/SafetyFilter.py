import numpy as np
from scipy.optimize import minimize
import time
from safety_filter.constraintQ import constraintQ
from safety_filter.constraintE import constraintE
from agent_training.constants import get_constants


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


def safety_filter(wheels_desired, t, state, n_F, theta_F, episode_count, episode_step): # gets control input, returns safe one

    initialize()
    global constants, outdata

    start = time.time()

    filter_log = ""
    
# Compute constraints
    H0, A0, b0 = constraintE(t, state, constants, 0, outdata)  # Energy constraint
    A1, b1, q_log = constraintQ(t, state, constants, 1, outdata, n_F, theta_F, episode_count, episode_step)  # Pointing constraint
    filter_log += q_log
    
# Setup QP optimization
    nonlcon_omega = lambda u: u.T @ H0 @ u + A0 @ u - b0  # Nonlinear energy constraint
    
    # Bounds for control input
    lower = -constants['u_max'] * np.ones(3)
    upper = constants['u_max'] * np.ones(3)
    
# Linear constraints: A*u <= b
    A = A1[np.newaxis, :]  # Shape (1, 3)
    b = np.array([b1])
    
    # Check if desired control satisfies constraints
    desired_clipped = np.clip(wheels_desired, lower, upper)
    linear_viol = A @ desired_clipped - b
    nonlin_viol = nonlcon_omega(desired_clipped)
    
    # If constraints already satisfied, return desired control
    if linear_viol <= 1e-6 and nonlin_viol <= 1e-6:
        outdata['u'] = desired_clipped
        outdata['compute'] = time.time() - start
        return desired_clipped
    
    try:
    # Quadratic objective: minimize ||u - wheels_desired||^2
        objective = lambda u: np.sum((u - wheels_desired)**2)
        
    # Gradient of objective for better convergence
        jac = lambda u: 2 * (u - wheels_desired)
        
    # Linear inequality constraint: A*u <= b (with small tolerance)
        linear_constraint = {'type': 'ineq', 
                           'fun': lambda u: (b - A @ u).flatten() + 1e-8}
        
    # Nonlinear inequality constraint: nonlcon_omega(u) <= 0
        # nonlin_constraint = {'type': 'ineq', 
        #                    'fun': lambda u: -nonlcon_omega(u) + 1e-8}
        
        # Use desired control as initial guess (clipped to bounds)
        u0 = desired_clipped
        
        result = minimize(objective, u0, method='SLSQP',
                        jac=jac,
                        bounds=[(lower[i], upper[i]) for i in range(3)],
                        constraints=[linear_constraint],
                        options={'disp': False, 'ftol': 1e-6, 'maxiter': 100})
        
        u_safe = result.x
        status = 1 if result.success else 0
        
    except Exception as e:
        print(f"Safety filter optimization error: {e}")
        filter_log += f"Safety filter optimization error: {e} at episode {episode_count}, step {episode_step}\n"
        u_safe = wheels_desired
        status = 0
    
    if status < 1:
        print(f"[{result.message},{episode_count},{episode_step}]",end=",")
        filter_log += f"{result.message} at episode {episode_count}, step {episode_step}\n"
        u_safe = wheels_desired
    
    if np.any(np.isnan(u_safe)):
        print('Warning: NaN values in safe control output')
        filter_log += f"NaN values in safe control output at episode {episode_count}, step {episode_step}\n"
        u_safe = wheels_desired
    
    # Store output data
    outdata['u'] = u_safe
    outdata['compute'] = time.time() - start

    #print(f"u_safe: {u_safe/7e-4}")
    
    return u_safe, filter_log
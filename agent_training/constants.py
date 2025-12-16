#This file is imported by SafetyFilter.py, constraintE.py, and constraintQ.py
import numpy as np

### PHYSICAL PARAMETERS
J_b = np.diag([0.1672, 0.1259, 0.06121])  # Body's moment of inertia (no wheels) [kg·m²] (diagonal 3x3)

J_w = 0.00001722  # One wheel's moment of inertia [kg·m²]

A=np.array([[0.0,  0.0,    0.8165, -0.8165],  # Wheel positions matrix (in body frame)
            [0.0, -0.9428, 0.4714,  0.4714],    # each collumn is one a_i, like in 2023 paper 
            [-1.0, 0.3333, 0.3333,  0.3333]])  

J_tot = J_b + J_w * A @ A.transpose()  # see 3 lines above eq. (2a) in 2023 paper 
# A @ A.T is a 3x3 matrix equal to the sum_i{(a_i)(a_i)T} (the sum of a_i's outer products)

u_max = 7e-4  # Max wheel torque [N·m]
w_max = 628.3 # Max wheel speed [rad/s] (for M1omega calculation)
xi_max = 1e-5 # Max disturbance torque [N·m]
e_max = 5.092e-5 # Max allowed kinetic energy [kg·m²/s²]


dt = 0.1  # Control timestep (how often do we update) [seconds]


### POINTING CONSTRAINT PARAMETERS

r_F = np.array([1, 0, 0]) # Boresight vector in body frame

# CBF parameters
#mu = ???        # Barrier function parameter
#delta_2 = ???   # Lower constraint threshold (lowercase delta)
#Delta_2 = ???   # Upper constraint threshold (uppercase Delta)
#M2plus = ???    # Second derivative bound
#M3plus = ???    # Third derivative bound


### COMPUTED PARAMETERS (Don't edit these, they're auto-calculated)

Jw_matrix = np.diag([J_w, J_w, J_w, J_w]) # turns scalar J_w value into 4x4 diagonal matrix
    
# Build Z matrix
top = np.hstack([J_tot, A @ Jw_matrix])              # top dimension   : (3x3) + (3x4)(4x4) = (3x3) + (3x4) = (3x7)
bottom = np.hstack([Jw_matrix @ A.T, Jw_matrix])    # bottom dimension: (4x4)(4x3) + (4x4) = (4x3) + (4x4) = (4x7)
M = np.vstack([top, bottom])                        # total dimension : (7x7) matrix 
Z = np.linalg.inv(M)                                # Z is also (7x7) as the inverse of M
    
Je = np.linalg.inv(Z[0:3, 0:3]) # used for constraintE
    
# Compute M1omega from wheel parameters
M1omega = 2 * w_max * u_max * np.sqrt(Je[0,0] / Je[2,2])    
M2omega = 1.95e-5 # Fixed value from paper


### EXPORT AS DICTIONARY
def get_constants(): # Returns a dictionary with all constants needed by the safety filter.
    
    constants = {
        # Computed dynamics
        'Z': Z,
        'Je': Je,
        
        # Energy constraint
        'e_max': e_max,
        #'M1omega': M1omega,
        'M2omega': M2omega,
        
        # Pointing constraint
        'r_F' : r_F,
        #'mu': mu,
        #'delta_2': delta_2,
        #'Delta_2': Delta_2,
        #'M2plus': M2plus,
        #'M3plus': M3plus,
        
        # Control parameters
        'dt': dt,
        'u_max': u_max,
        
        # Physical parameters (for reference)
        'J_tot': J_tot,
        'A': A,
        'J_w': Jw_matrix
    }
    
    return constants

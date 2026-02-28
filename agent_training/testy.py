import numpy as np
import matplotlib.pyplot as plt
import csv
import math

min_angle = 0.0
max_angle = 40.0
scale_parameter = (max_angle-min_angle)*0.5  # scale parameter for exponential distribution

samples = []

# load csv file and read the initial error angles
# with open('drl_reorientation/agent_training/test.json', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     next(csvreader)  # skip header
#     for row in csvreader:
#         initial_error_angle = float(row[3])  # assuming the 4th column is initial_error_angle
#         samples.append(initial_error_angle)

# for _ in range(1000000):
#     if min_angle == max_angle:
#         samples.append(min_angle)
#         continue
#     sample = np.random.exponential(scale_parameter)
#     sample = max_angle - sample # inverse distribution direction
    
    
#     while sample < min_angle or sample > max_angle:
#         sample = np.random.exponential(scale_parameter)
#         sample = max_angle - sample # inverse distribution direction

#     samples.append(sample)

    

# shift and clip the exponential samples to be within the desired range
#samples_exponential = np.clip(samples_exponential + min_angle, min_angle, max_angle)

# plot histogram of the samples
# plt.hist(samples, bins=50, density=True, alpha=0.7, color='blue')
# plt.title('Histogram of Exponentially Distributed Random Samples')
# plt.xlabel('Angle')
# plt.ylabel('Density')
# plt.grid(True)
# plt.show()


#########################
scale_torque = 7e-4
scale_torque_norm = np.sqrt(scale_torque**2 + scale_torque**2 + scale_torque**2)

torque_1 = scale_torque * 1
torque_2 = scale_torque * 1
torque_3 = scale_torque * 1
# torque_1_prev = scale_torque * -1
# torque_2_prev = scale_torque * -1
# torque_3_prev = scale_torque * -1

# rw_torque = - 0.05*(math.sqrt(torque_1**2 + torque_2**2 + torque_3**2)/scale_torque_norm)
# rw_freq_torque = - 0.005*2860.0*math.sqrt((torque_1 - torque_1_prev)**2 + (torque_2 - torque_2_prev)**2 + (torque_3 - torque_3_prev)**2)

# print("rw_torque:", rw_torque)
# print("rw_freq_torque:", rw_freq_torque)

res1 = np.linalg.norm([torque_1, torque_2, torque_3])
torque_1 = scale_torque * 1
torque_2 = scale_torque * 0
torque_3 = scale_torque * -1

res2 = np.linalg.norm([torque_1, torque_2, torque_3])
torque_1 = scale_torque * -1
torque_2 = scale_torque * -1
torque_3 = scale_torque * -1

res3 = np.linalg.norm([torque_1, torque_2, torque_3])

print("res1:", res1)
print("res2:", res2)
print("res3:", res3)

print("-----------")

action1 = np.array([1.0, 0.0, 0.0])

action_st = action1

action2 = np.array([0.0, 1.0, 0.0])
action1 = action2

print(action_st)
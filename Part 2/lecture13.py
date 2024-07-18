# ## This is course material for Introduction to Python Scientific Programming
# ## Example code: matplotlib_clock.py
# ## Author: Allen Y. Yang
# ##
# ## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import time

# # Initialization, define some constant
# path = os.path.dirname(os.path.abspath(__file__))
# filename = path + '/airplane.bmp'
# background = plt.imread(filename)

# second_hand_length = 200
# second_hand_width = 2
# minute_hand_length = 150
# minute_hand_width = 6
# hour_hand_length = 100
# hour_hand_width = 10
# gmt_hand_length = 100
# gmt_hand_width = 10
# center = np.array([256, 256])

# def clock_hand_vector(angle, length):
#     return np.array([length * np.sin(angle), -length * np.cos(angle)])

# # draw an image background
# fig, ax = plt.subplots()
# while True:
#     plt.imshow(background)
#     # First retrieve the time
#     now_time = datetime.now()
#     hour = now_time.hour
#     if hour>12: hour = hour - 12
#     minute = now_time.minute
#     second = now_time.second
#     gmt = time.gmtime

#     # Calculate end points of hour, minute, second

#     hour_vector = clock_hand_vector(hour/12*2*np.pi, hour_hand_length)
#     minute_vector = clock_hand_vector(minute/60*2*np.pi, minute_hand_length)
#     second_vector = clock_hand_vector(second/60*2*np.pi, second_hand_length)
#     gmt_vector = clock_hand_vector(gmt/12*2*np.pi, gmt_hand_length)

#     plt.arrow(center[0], center[1], gmt_vector[0], gmt_vector[1], head_length = 3, linewidth = gmt_hand_width, color = 'yellow')
#     plt.arrow(center[0], center[1], hour_vector[0], hour_vector[1], head_length = 3, linewidth = hour_hand_width, color = 'black')
#     plt.arrow(center[0], center[1], minute_vector[0], minute_vector[1], linewidth = minute_hand_width, color = 'black')
#     plt.arrow(center[0], center[1], second_vector[0], second_vector[1], linewidth = second_hand_width, color = 'red')
    
#     plt.axis('off')

#     plt.pause(0.1)
#     plt.clf()




## This is course material for Introduction to Python Scientific Programming
## Example code: gradient_descent.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()

sample_count = 100
x_sample = 10*np.random.random(sample_count)-5
y_sample = 2*x_sample - 1 + np.random.normal(0, 1.0, sample_count)

# plots the parameter space
ax2 = fig.add_subplot(1,1,1, projection = '3d')

def penalty(para_a, para_b):
    global x_sample, y_sample, sample_count

    squares = (y_sample - para_a*x_sample - para_b)**2
    return 1/2/sample_count*np.sum(squares)

a_arr, b_arr = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1) )

func_value = np.zeros(a_arr.shape)
for x in range(a_arr.shape[0]):
    for y in range(a_arr.shape[1]):
            func_value[x, y] = penalty(a_arr[x, y], b_arr[x, y])

ax2.plot_surface(a_arr, b_arr, func_value, color = 'red', alpha = 0.8)
ax2.set_xlabel('a parameter')
ax2.set_ylabel('b parameter')
ax2.set_zlabel('f(a, b)')

# Plot the gradient descent
def grad(aa):
    grad_aa = np.zeros(2)
    update_vector = (y_sample - aa[0] * x_sample - aa[1])
    grad_aa[0] = - 1/sample_count * x_sample.dot(update_vector)
    grad_aa[1] = - 1/sample_count * np.sum(update_vector)
    return grad_aa

aa = np.array([-4, 4])
delta = np.inf
epsilon = 0.001
learn_rate = 0.2
step_count = 0
ax2.scatter(aa[0], aa[1], penalty(aa[0],aa[1]), c='b', s=100, marker='*')
# Update vector aa
while delta > epsilon:
    # Gradient Descent
    aa_next = aa - learn_rate * grad(aa)
    # Plot the animation
    ax2.plot([aa[0],aa_next[0]],[aa[1], aa_next[1]],\
        [penalty(aa[0],aa[1]), penalty(aa_next[0],aa_next[1]) ], 'ko-')
    delta = np.linalg.norm(aa - aa_next)
    aa = aa_next
    step_count +=1
    fig.canvas.draw_idle()
    plt.pause(0.1)

print('Optimal result: ', aa)
ax2.scatter(aa[0], aa[1], penalty(aa[0],aa[1]), c='b', s=100, marker='*')
plt.show()
print('Step Count:', step_count)
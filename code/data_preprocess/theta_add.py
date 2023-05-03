import numpy as np
import torch
from math import pi


def theta_trans(trajectory):
    theta_add_trajectory = np.zeros(trajectory.shape)
    theta_add_trajectory[0] = trajectory[0]
    for i in range(1, len(trajectory)):
        theta_add_trajectory[i, [0, 1, 3]] = trajectory[i, [0, 1, 3]]
        if trajectory[i - 1, 2] - trajectory[i, 2] > pi:
            theta_add_trajectory[i, 2] = theta_add_trajectory[i - 1, 2] + 2 * pi + trajectory[i, 2] - trajectory[
                i - 1, 2]
        elif trajectory[i - 1, 2] - trajectory[i, 2] < -pi:
            theta_add_trajectory[i, 2] = theta_add_trajectory[i - 1, 2] + (
                    trajectory[i, 2] - 2 * pi - trajectory[i - 1, 2])
        else:
            theta_add_trajectory[i, 2] = theta_add_trajectory[i - 1, 2] + trajectory[i, 2] - trajectory[i - 1, 2]
    return theta_add_trajectory

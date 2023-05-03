import numpy as np


# this function is used to change the time of each points of recorded trajectory to relative time of 1st point
def change_time_relative(trajectory):
    for point in trajectory[1:]:
        point[-1] = point[-1] - trajectory[0][-1]
    trajectory[0][-1] = 0
    return trajectory
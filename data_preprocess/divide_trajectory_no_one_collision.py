import sys
import numpy as np
from theta_add import theta_trans
import matplotlib.pyplot as plt
import torch
import tqdm


table_length = 1.948
puck_radius = 0.03165
table_width = 1.038

def close_boundary(point):
    if point[0] < puck_radius + 0.05 or point[0] > (table_length - puck_radius - 0.05) or point[1] < (
            -table_width / 2 + puck_radius + 0.05) or point[1] > (table_width / 2 - puck_radius - 0.05):
        return True
    return False


def has_collision(pre, cur, next):
    if (next[0] - cur[0]) * (cur[0] - pre[0]) < 0 or (next[1] - cur[1]) * (cur[1] - pre[1]) < 0:
        return True
    return False


def trajectory_has_collision(trajectory):
    collision_pos_set = []
    for i in range(1, len(trajectory) - 1):
        if has_collision(trajectory[i - 1], trajectory[i], trajectory[i + 1]) and close_boundary(trajectory[i]):
            collision_pos_set.append(i)
    if collision_pos_set != []:
        return True, collision_pos_set
    else:
        return False, collision_pos_set


if __name__ == '__main__':
    sys.path.append('/home/dhr/RLIP-EKF/program/')
    # code can run, ignore
    from cov_dyna_params import generate_params
    from torch_EKF_Wrapper import AirHockeyEKF
    from total_test_plot_other_use import calculate_init_state
    save = False
    theta_sum = True
    device = torch.device("cpu")
    result = np.load('../hundred_trajectories.npy', allow_pickle=True)
    result_clean = [[] for i in range(len(result))]
    judge_collision = 'related_movement' # kinematics  related_movement
    # clean the data, remove still part; if recording gap too long, remove too
    for i in range(len(result)):

        for j in range(1, len(result[i])):
            if abs(result[i][j][0] - result[i][j - 1][0]) < 0.005 and abs(
                    result[i][j][1] - result[i][j - 1][1]) < 0.005:
                continue
            if result_clean[i] != []:
                if result[i][j][3] - result_clean[i][-1][3] > 5 / 120:
                    break
            result_clean[i].append(result[i][j])
        if i != 79:
            result_clean[i] = np.vstack(result_clean[i])
    result_clean = np.array(result_clean)
    reduce_num = 0
    for index, trajectory in enumerate(result_clean):
        if len(trajectory) < 15:
            result_clean = np.delete(result_clean, index - reduce_num)
            reduce_num += 1
    # move to table left middle coordinate
    for i in range(len(result_clean)):
        for i_data in result_clean[i]:
            i_data[0] += table_length / 2
    # change theta to no limit of -pi, pi
    if theta_sum:
        for index, trajectory in enumerate(result_clean):
            trajectory = theta_trans(trajectory)
    # if save:
    #     np.save('hundred_data_after_clean', result_clean)
    P, R, Q, Q_collision, dyna_params, dynamic_system = generate_params(device)
    puck_EKF = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=device,
                            Q_collision=Q_collision)
    one_collision_set = []
    no_collision_set = []
    # kinematics way to get collision point index
    if judge_collision == 'kinematics':
        for trajectory in result_clean:
            tensor_trajectory = torch.tensor(trajectory, device=device).float()
            init_state = calculate_init_state(trajectory, device=device)
            filter_result = puck_EKF.kalman_filter(init_state, tensor_trajectory[1:])
            coll_index = np.where(filter_result[1])[0]
            if len(coll_index) == 0:
                no_collision_set.append(trajectory)
            elif len(coll_index) == 1:
                one_collision_set.append(trajectory)
                no_collision_set.append(trajectory[0:coll_index[0]] + 1)
                no_collision_set.append(trajectory[coll_index[0] + 1:])
            else:
                no_collision_set.append(trajectory[0:coll_index[0] + 1])
                for i in range(len(coll_index) - 1):
                    if coll_index[i+1] - coll_index[i] > 15:
                        no_collision_set.append(trajectory[coll_index[i]+1: coll_index[i+1] + 1])
                if len(trajectory) - coll_index[i + 1] > 15:
                    no_collision_set.append(trajectory[coll_index[i+1] + 1:])
    # related movement to get collision point index
    elif judge_collision == 'related_movement':
        for trajectory in result_clean:
            whether_collision, coll_index = trajectory_has_collision(trajectory)
            if len(coll_index) == 0:
                if len(trajectory) > 15:
                    no_collision_set.append(trajectory)
            elif len(coll_index) == 1:
                if coll_index[0] > 10:
                    one_collision_set.append(trajectory)
                if coll_index[0] > 15:
                    no_collision_set.append(trajectory[0:coll_index[0]])
                if len(trajectory) - coll_index[0] > 15:
                    no_collision_set.append(trajectory[coll_index[0] + 2:])
            else:
                if coll_index[0] > 15:
                    no_collision_set.append(trajectory[0:coll_index[0]])
                for i in range(len(coll_index) - 1):
                    if coll_index[i+1] - coll_index[i] > 15:
                        no_collision_set.append(trajectory[coll_index[i]+1: coll_index[i+1]])
                if len(trajectory) - coll_index[i + 1] > 15:
                    no_collision_set.append(trajectory[coll_index[i+1] + 1:])
                # divide trajectory into one-collision trajectory
                if coll_index[1] - coll_index[0] > 10 and coll_index[0] > 10:
                    one_collision_set.append(trajectory[0:coll_index[1] - 1])
                for i in range(len(coll_index)):
                    if i == len(coll_index) - 1:
                        continue
                    if i == len(coll_index) - 2 and len(trajectory) - coll_index[i + 1] > 10 and coll_index[i + 1] - coll_index[i] > 10:
                        one_collision_set.append(trajectory[coll_index[i] + 1:])
                    elif i < len(coll_index) - 2 and coll_index[i + 2] - coll_index[i + 1] > 10 and coll_index[i + 1] - coll_index[i] > 10:
                        one_collision_set.append(trajectory[coll_index[i] + 1: coll_index[i + 2] - 1])
    del(one_collision_set[173])
    if save:
        np.save('hundred_data_one_coll', one_collision_set)
        np.save('hundred_data_no_coll', no_collision_set)

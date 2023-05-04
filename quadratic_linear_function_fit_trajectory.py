import numpy as np
from data_preprocess.time_clear import change_time_relative
import matplotlib.pyplot as plt
from torch_gradient import judge_time_alignment
from data_preprocess.theta_add import theta_trans
from cov_dyna_params import generate_params
import torch
from torch_EKF_Wrapper import AirHockeyEKF
from torch_air_hockey_baseline_no_detach import SystemModel


def calculate_init_state(trajectory, device=torch.device("cpu"), type='interpolate'):
    if type == 'interpolate':
        dx = ((trajectory[1][0] - trajectory[0][0]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][0] - trajectory[1][0]) / (
                      trajectory[2][3] - trajectory[1][3]) + (trajectory[3][0] - trajectory[2][0]) / (
                      trajectory[3][3] - trajectory[2][3])) / 3
        dy = ((trajectory[1][1] - trajectory[0][1]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][1] - trajectory[1][1]) / (
                      trajectory[2][3] - trajectory[1][3]) + (trajectory[3][1] - trajectory[2][1]) / (
                      trajectory[3][3] - trajectory[2][3])) / 3
        dtheta = ((trajectory[1][2] - trajectory[0][2]) / (trajectory[1][3] - trajectory[0][3]) + (
                trajectory[2][2] - trajectory[1][2]) / (
                          trajectory[2][3] - trajectory[1][3]) + (trajectory[3][2] - trajectory[2][2]) / (
                          trajectory[3][3] - trajectory[2][3])) / 3
        state_ = torch.tensor([trajectory[1][0], trajectory[1][1], dx, dy, trajectory[1][2], dtheta],
                              device=device).float()
    return state_


# use quadratic function to fit
def quadratic_fit(trajectory):
    trajectory = change_time_relative(trajectory)
    x_params = np.polyfit(trajectory[:, 3], trajectory[:, 0], 2)
    y_params = np.polyfit(trajectory[:, 3], trajectory[:, 1], 2)
    theta_params = np.polyfit(trajectory[:, 3], trajectory[:, 2], 2)
    return x_params, y_params, theta_params


# use fitted parameters and derivation to get velocity
def get_velocity(trajectory, x_params, y_params, theta_params):
    state_list = np.zeros((len(trajectory), 6))
    for index, point in enumerate(trajectory):
        state_list[index, 0] = x_params[0] * point[3] ** 2 + x_params[1] * point[3] + x_params[2]
        state_list[index, 1] = y_params[0] * point[3] ** 2 + y_params[1] * point[3] + y_params[2]
        state_list[index, 4] = theta_params[0] * point[3] ** 2 + theta_params[1] * point[3] + theta_params[2]
        # state_list[index, [0, 1, 4]] = trajectory[index, 0:3]
        state_list[index, 2] = 2 * x_params[0] * point[3] + x_params[1]
        state_list[index, 3] = 2 * y_params[0] * point[3] + y_params[1]
        state_list[index, 5] = 2 * theta_params[0] * point[3] + theta_params[1]
    return state_list


# it can be used for calculate non collision / collision dynamic variance
def calculate_variance(potential_true_trajectory, predict_trajectory):
    Q = np.eye(6)
    innovation = np.vstack(potential_true_trajectory) - np.vstack(predict_trajectory)
    for i in range(6):
        Q[i, i] = np.var(innovation[:, i])
    return Q


def calculate_coll_variance(input, output):
    P, R, Q, Q_collision, dyna_params, dynamic_system = generate_params(device)
    dynamic_system = SystemModel(tableDampingX=dyna_params[0], tableDampingY=dyna_params[1],
                                 tableFrictionX=dyna_params[2],
                                 tableFrictionY=dyna_params[3],
                                 tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                 puckRadius=0.03165, malletRadius=0.04815,
                                 tableRes=dyna_params[4],
                                 malletRes=0.8, rimFriction=dyna_params[5],
                                 dt=1 / 120, device=device)
    predict_list = []
    local_output_list = []
    local_input_List = []
    for index in range(len(input)):
        input_state = torch.from_numpy(input[index]).type(torch.FloatTensor).to(device)
        output_state = torch.from_numpy(output[index]).type(torch.FloatTensor).to(device)
        predict_state = dynamic_system.apply_collision(input_state, coll_mode=True, cal_mode='statistik')
        if (not predict_state[0]) or (predict_state[-3] @ output_state)[3] < 0:
            continue
        print(predict_state[0])
        predict_list.append((predict_state[-3] @ predict_state[1]).numpy())
        local_output_list.append((predict_state[-3] @ output_state).numpy())
        local_input_List.append((predict_state[-3] @ input_state).numpy())
    predict_list = np.vstack(predict_list)
    local_output_list = np.vstack(local_output_list)
    return np.cov((local_output_list - predict_list).T)


def linear_regression_non_collision_params(potential_true_state):
    input_state = []
    output_state = []
    for trajectory in potential_true_state:
        input_state.append(trajectory[0:len(trajectory) - 1])
        output_state.append(trajectory[1:])
    input_state = np.vstack(input_state)
    output_state = np.vstack(output_state)
    new_damping_x = 120 * input_state[:, 2] @ (input_state[:, 2] - output_state[:, 2]) / (
            input_state[:, 2] @ input_state[:, 2])
    new_damping_y = 120 * input_state[:, 3] @ (input_state[:, 3] - output_state[:, 3]) / (
            input_state[:, 3] @ input_state[:, 3])
    return new_damping_x, new_damping_y


def linear_regression_collision_params(collision_input, collision_output, collision_slide_input, collision_slide_output,
                                       slide_list):
    collision_input = np.vstack(collision_input)
    collision_output = np.vstack(collision_output)
    collision_slide_input = np.vstack(collision_slide_input)
    collision_slide_output = np.vstack(collision_slide_output)
    restitution = -collision_input[:, 3] @ collision_output[:, 3] / (collision_input[:, 3] @ collision_input[:, 3])
    b_diff = (collision_slide_output[:, 2] - collision_slide_input[:, 2]) * slide_list
    # 0.03165 is puck radius
    theta_diff = (collision_slide_output[:, 5] - collision_slide_input[:, 5]) * slide_list * 0.03165 / 2
    n_input = np.append(collision_slide_input[:, 3], collision_slide_input[:, 3])
    friction = np.append(b_diff, theta_diff) @ n_input / ((n_input @ n_input) * (1 + restitution))
    return friction, restitution


# check the collision mode, slide or not,
# input velocity before and after collision
def check_collision_mode(input_state, output_state):
    predict_b = 2 * input_state[2] / 3 - 0.03165 * input_state[5] / 3
    if abs((predict_b - output_state[2]) / output_state[2]) < 0.2:
        return 'no_slide'
    else:
        return 'slide'


if __name__ == '__main__':
    plot = False
    save = False
    non_coll_dataset = np.load('./data/hundred_data_no_coll.npy', allow_pickle=True)
    one_coll_dataset = np.load('./data/hundred_data_one_coll.npy', allow_pickle=True)
    device = torch.device("cpu")
    # used for linear regression calculating non collision dynamic parameters
    non_coll_input = []
    non_coll_output = []
    non_coll_record = []
    non_coll_potential_true = []
    non_coll_predict = []
    non_coll_dt = []
    # calculate variance for non collision part
    # try using EKF code to do one step predict
    for trajectory in non_coll_dataset:
        if not judge_time_alignment(trajectory):
            continue
        trajectory = theta_trans(trajectory)
        x_params, y_params, theta_params = quadratic_fit(trajectory)
        state_list = get_velocity(trajectory, x_params, y_params, theta_params)
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.scatter(trajectory[:, -1], trajectory[:, 0], label='record', c='b')
        # plt.scatter(trajectory[:, -1], state_list[:, 0], label='quadratic', c='r')
        # plt.subplot(3, 1, 2)
        # plt.scatter(trajectory[:, -1], trajectory[:, 1], label='record', c='b')
        # plt.scatter(trajectory[:, -1], state_list[:, 1], label='quadratic', c='r')
        # plt.subplot(3, 1, 3)
        # plt.scatter(trajectory[:, -1], trajectory[:, 2], label='record', c='b')
        # plt.scatter(trajectory[:, -1], state_list[:, 4], label='quadratic', c='r')
        # plt.legend()
        non_coll_record.append(trajectory)
        non_coll_potential_true.append(state_list)
        non_coll_input.append(state_list[0:len(state_list) - 1])
        non_coll_output.append(state_list[1:])
        non_coll_dt.append(trajectory[1:, 3] - trajectory[0:len(trajectory) - 1, 3])
    # calculate variance for dynamic part
    # plt.show()
    potential = np.vstack(non_coll_potential_true)
    record = np.vstack(non_coll_record)
    innovation = potential[:, [0, 1, 4]] - record[:, 0:3]
    R = torch.eye(3, device=device)
    for i in range(3):
        R[i, i] = np.var(innovation[:, i])
    print('R' + str(R))
    non_coll_input = np.vstack(non_coll_input)
    non_coll_output = np.vstack(non_coll_output)
    P, R, Q, Q_collision, dyna_params, dynamic_system = generate_params(device)
    puck_EKF = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=device,
                            Q_collision=Q_collision)
    for potential_true in non_coll_potential_true:
        tensor_potential_true = torch.from_numpy(potential_true).type(torch.FloatTensor)
        tensor_predict = puck_EKF.smooth_one_step_predict(tensor_potential_true)
        predict_list = [item.numpy() for item in tensor_predict]
        predict_list = np.vstack(predict_list)
        non_coll_predict.append(predict_list)
    Q = calculate_variance(non_coll_potential_true, non_coll_predict)
    print('Q' + str(Q))
    new_damping_x, new_damping_y = linear_regression_non_collision_params(non_coll_potential_true)
    ##################################################################
    ##################################################################
    # calculate variance for collision part
    coll_slide_observation = []  # record data
    coll_no_slide_observation = []  # record data
    coll_slide_input = []
    coll_slide_output = []
    coll_no_slide_input = []
    coll_no_slide_output = []
    global_input = []
    global_output = []
    s_list = []  # record slide collision s list
    s_no_slide_list = []
    count = 0
    for index, trajectory in enumerate(one_coll_dataset):
        # if not judge_time_alignment(trajectory):
        #     continue
        trajectory = theta_trans(trajectory)
        tensor_trajectory = torch.from_numpy(trajectory).type(torch.FloatTensor)
        init_state = calculate_init_state(trajectory, device=device)
        filter_result = puck_EKF.kalman_filter(init_state, tensor_trajectory[1:], coll_mode=True, cal_mode='statistik')
        coll_index = filter_result[1]
        coll_index = np.where(coll_index)
        if (len(coll_index[0]) == 0):
            continue
        count += 1
        pre_coll_trajectory = trajectory[0:int(coll_index[0][0]) + 1]
        post_coll_trajectory = trajectory[int(coll_index[0][0]) + 1:]
        pre_x_params, pre_y_params, pre_theta_params = quadratic_fit(pre_coll_trajectory)
        pre_coll_state_list = get_velocity(pre_coll_trajectory, pre_x_params, pre_y_params, pre_theta_params)
        post_x_params, post_y_params, post_theta_params = quadratic_fit(post_coll_trajectory)
        post_coll_state_list = get_velocity(post_coll_trajectory, post_x_params, post_y_params, post_theta_params)
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.scatter(trajectory[:, -1], trajectory[:, 0], label='record', c='b')
        # plt.scatter(trajectory[0:int(coll_index[0][0]) + 1, -1], pre_coll_state_list[:, 0], label='quadratic pre',
        #             c='r', alpha=0.2)
        # plt.scatter(trajectory[int(coll_index[0][0]) + 1:, -1], post_coll_state_list[:, 0], label='quadratic post',
        #             c='k', alpha=0.2)
        # plt.legend()
        # plt.subplot(3, 1, 2)
        # plt.scatter(trajectory[:, -1], trajectory[:, 1], label='record', c='b')
        # plt.scatter(trajectory[0:int(coll_index[0][0]) + 1, -1], pre_coll_state_list[:, 1], label='quadratic pre',
        #             c='r', alpha=0.2)
        # plt.scatter(trajectory[int(coll_index[0][0]) + 1:, -1], post_coll_state_list[:, 1], label='quadratic post',
        #             c='k', alpha=0.2)
        # plt.subplot(3, 1, 3)
        # plt.scatter(trajectory[:, -1], trajectory[:, 2], label='record', c='b')
        # plt.scatter(trajectory[0:int(coll_index[0][0]) + 1, -1], pre_coll_state_list[:, 4], label='quadratic pre',
        #             c='r', alpha=0.2)
        # plt.scatter(trajectory[int(coll_index[0][0]) + 1:, -1], post_coll_state_list[:, 4], label='quadratic post',
        #             c='k', alpha=0.2)

        collision_mode = check_collision_mode(
            filter_result[-5][int(coll_index[0][0])].numpy() @ pre_coll_state_list[-1],
            filter_result[-5][int(coll_index[0][0])].numpy() @ post_coll_state_list[0])
        if collision_mode == 'slide':
            # if filter_result[-4][int(coll_index[0][0])] == 'slide':
            s_list.append(int(filter_result[-3][int(coll_index[0][0])]))
            observation_xy = filter_result[-5][int(coll_index[0][0])].numpy()[0:2, 0:2] @ trajectory[int(
                coll_index[0][0]) + 1][0:2]
            observation_theta_t = trajectory[int(coll_index[0][0]) + 1][2:]
            observation = np.append(observation_xy, observation_theta_t)
            coll_slide_observation.append(observation)
            coll_slide_input.append(filter_result[-5][int(coll_index[0][0])].numpy() @ pre_coll_state_list[-1])
            coll_slide_output.append(filter_result[-5][int(coll_index[0][0])].numpy() @ post_coll_state_list[0])
        elif collision_mode == 'no_slide':
            # elif filter_result[-4][int(coll_index[0][0])] == 'no_slide':
            s_no_slide_list.append(int(filter_result[-3][int(coll_index[0][0])]))
            observation_xy = filter_result[-5][int(coll_index[0][0])].numpy()[0:2, 0:2] @ trajectory[int(
                coll_index[0][0]) + 1][0:2]
            observation_theta_t = trajectory[int(coll_index[0][0]) + 1][2]
            observation = np.append(observation_xy, observation_theta_t)
            coll_no_slide_observation.append(observation)
            coll_no_slide_input.append(filter_result[-5][int(coll_index[0][0])].numpy() @ pre_coll_state_list[-1])
            coll_no_slide_output.append(filter_result[-5][int(coll_index[0][0])].numpy() @ post_coll_state_list[0])
        global_input.append(pre_coll_state_list[-1])
        global_output.append(post_coll_state_list[0])
    plt.show()
    coll_observation = coll_slide_observation + coll_no_slide_observation
    coll_input = coll_slide_input + coll_no_slide_input
    coll_output = coll_slide_output + coll_no_slide_output
    Q_collision = calculate_coll_variance(np.vstack(global_input), np.vstack(global_output))
    print('Q_collision' + str(Q_collision))
    new_friction, new_restitution = linear_regression_collision_params(coll_input, coll_output, coll_slide_input,
                                                                       coll_slide_output, s_list)
    if save:
        np.save('s_slide_list', s_list)
        np.save('s_no_slide_list', s_no_slide_list)
        np.save('coll_input', coll_input)
        np.save('coll_output', coll_output)
    if plot:
        # plt.figure()
        # plt.scatter(np.vstack(coll_input)[:, 2], np.vstack(coll_output)[:, 2], c='b', label='b direction', alpha=0.2)
        # plt.legend()
        plt.figure()
        plt.scatter(np.vstack(coll_input)[:, 3], np.vstack(coll_output)[:, 3], c='b', label='quadratic fit', alpha=0.2)
        plt.scatter(np.vstack(coll_input)[:, 3], -0.786 * np.vstack(coll_input)[:, 3], c='r', label='multy restitution',
                    alpha=0.2)
        plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(np.vstack(coll_slide_input)[:, 2], np.vstack(coll_slide_input)[:, 3] * s_list,
                      np.vstack(coll_slide_output)[:, 2], label='n slide', c='b', alpha=0.2)
        ax1.scatter3D(np.vstack(coll_no_slide_input)[:, 2], np.vstack(coll_no_slide_input)[:, 3] * s_no_slide_list,
                      np.vstack(coll_no_slide_output)[:, 2], label='n no slide', c='r', alpha=0.2)

        ax1.set_xlabel('b', fontsize=20)
        ax1.set_ylabel('n', fontsize=20)
        ax1.set_zlabel('b_out', fontsize=20)
        ax1.legend()
        plt.figure()
        ax2 = plt.axes(projection='3d')
        ax2.scatter3D(np.vstack(coll_slide_input)[:, 5], np.vstack(coll_slide_input)[:, 2],
                      np.vstack(coll_slide_output)[:, 2], label='n slide', c='b', alpha=0.2)
        ax2.scatter3D(np.vstack(coll_no_slide_input)[:, 5], np.vstack(coll_no_slide_input)[:, 2],
                      np.vstack(coll_no_slide_output)[:, 2], label='n no slide', c='r', alpha=0.2)

        ax2.set_xlabel('theta', fontsize=20)
        ax2.set_ylabel('b', fontsize=20)
        ax2.set_zlabel('b_out', fontsize=20)
        ax2.legend()

    plt.show()

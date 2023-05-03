import matplotlib.pyplot as plt
from torch_EKF_Wrapper import AirHockeyEKF
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
import numpy as np
import torch
from statedependentparams import FullResState
from statedependentparams import ResState
import sys
sys.path.append('/home/dhr/RLIP-EKF/program/data_preprocess/')
from divide_trajectory_no_one_collision import trajectory_has_collision
import quadratic_linear_function_fit_trajectory
from  data_preprocess.theta_add import theta_trans


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
    elif type == 'fit':
        has_collision, coll_index = trajectory_has_collision(trajectory)
        if has_collision:
            trajectory = trajectory[0:coll_index[0] - 2]
        x_params, y_params, theta_params = quadratic_linear_function_fit_trajectory.quadratic_fit(trajectory)
        state_list = quadratic_linear_function_fit_trajectory.get_velocity(trajectory, x_params, y_params, theta_params)
        state_ = torch.from_numpy(state_list[0]).type(torch.FloatTensor).to(device)
    return state_


if __name__ == '__main__':
    data_file_name = 'hundred_data_one_coll.npy'
    model_file_name = './alldata/EKF+EKF+multi_mse+only_collision_nn+2mode_collision+1106/model.pt'
    device = torch.device("cpu")
    full_res = None
    plot = True
    # full_res = FullResState(device=device)
    # full_res.to(device)
    # full_res.load_state_dict(torch.load(model_file_name))
    # res = ResState(device=device)
    # res.to(device)
    # res.load_state_dict(torch.load(model_file_name))
    res = None
    total_dataset = np.load(data_file_name, allow_pickle=True)

    # covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 25e-6, 25e-2, 0.0225, 225]).to(device=device)
    # covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 4e-10, 4e-6, 1e-6, 0.01]).to(device=device)
    covariance_params_collision = torch.Tensor([6.4e-5, 6.4e-5, 9.1e-3, 1e-4, 1, 0.0225, 225]).to(device=device)
    covariance_params = torch.Tensor([6.4e-5, 6.4e-5, 9.1e-3, 4e-10, 4e-6, 1e-6, 0.01]).to(device=device)
    # params: damping x, damping y, friction x, friction y, restitution, rimfriction
    dyna_params_gradient = torch.tensor([0.2, 0.2, 0.01, 0.01, 0.798, 0.122], device=device)
    # EKF linear regression
    # dyna_params = torch.tensor([0.191, 0.212, 0.01, 0.01, 0.782, 0.1176], device=device)
    # original params from puze
    # dyna_params = torch.tensor([0.001, 0.001, 0.001, 0.001, 0.7424, 0.1418], device=device)
    # quadratic function fit and linear regression
    dyna_params = torch.tensor([0.2125, 0.2562, 0.01, 0.01, 0.786, 0.098072], device=device)
    R = torch.diag(torch.stack([covariance_params[0],
                                covariance_params[1],
                                covariance_params[2]]))
    Q = torch.diag(torch.stack([covariance_params[3], covariance_params[3],
                                covariance_params[4], covariance_params[4],
                                covariance_params[5], covariance_params[6]]))
    Q_collision = torch.diag(
        torch.stack([covariance_params_collision[3], covariance_params_collision[3],
                     covariance_params_collision[4], covariance_params_collision[4],
                     covariance_params_collision[5], covariance_params_collision[6]]))
    R = np.load('./observation_variance.npy')
    Q = np.load('./dynamic_no_coll_var.npy')
    Q_collision = np.load('./dynamic_coll_var.npy')
    R = torch.from_numpy(R).type(torch.FloatTensor).to(device)
    Q = torch.from_numpy(Q).type(torch.FloatTensor).to(device)
    Q_collision = torch.from_numpy(Q_collision).type(torch.FloatTensor).to(device)
    P = torch.eye(6, device=device) * 0.01
    dynamic_system = torch_air_hockey_baseline.SystemModel(tableDampingX=dyna_params[0],
                                                           tableDampingY=dyna_params[1],
                                                           tableFrictionX=dyna_params[2],
                                                           tableFrictionY=dyna_params[3],
                                                           tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                           puckRadius=0.03165, malletRadius=0.04815,
                                                           tableRes=dyna_params[4],
                                                           malletRes=0.8, rimFriction=dyna_params[5],
                                                           dt=1 / 120, device=device)
    puck_EKF = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=device,
                            Q_collision=Q_collision)
    dynamic_gradient_system = torch_air_hockey_baseline.SystemModel(tableDampingX=dyna_params_gradient[0],
                                                                    tableDampingY=dyna_params_gradient[1],
                                                                    tableFrictionX=dyna_params_gradient[2],
                                                                    tableFrictionY=dyna_params_gradient[3],
                                                                    tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                                    puckRadius=0.03165, malletRadius=0.04815,
                                                                    tableRes=dyna_params_gradient[4],
                                                                    malletRes=0.8, rimFriction=dyna_params_gradient[5],
                                                                    dt=1 / 120, device=device)
    puck_EKF_gradient = AirHockeyEKF(u=1 / 120., system=dynamic_gradient_system, Q=Q, R=R, P=P, device=device,
                                     Q_collision=Q_collision)
    puck_EKF_statistik = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=device,
                                      Q_collision=Q_collision)
    dynamic_innovation_list = []
    observation_innovation_list = []
    # for i in range(80, len(total_dataset)):

    # for i in range(len(total_dataset)):
    for i in range(0, 30):
        trajectory = total_dataset[i]
        trajectory = theta_trans(trajectory)
        trajectory_tensor = torch.tensor(trajectory, device=device).float()
        init_state = calculate_init_state(trajectory, device=device, type='fit')
        # filter_state, _, _, _, _, predict_state = puck_EKF_gradient.kalman_filter(init_state, trajectory_tensor[1:],
        #                                                                           full_res=full_res)
        # filter_state = torch.vstack(filter_state)
        # predict_state = torch.vstack(predict_state)
        # if len(filter_state[:, [0, 1]]) == len(trajectory_tensor[1:, 0:2]):
        #     dynamic_innovation_list.append(filter_state[:, [0, 1]] - trajectory_tensor[1:, 0:2])
        #     observation_innovation_list.append(filter_state[:, [0, 1]] - predict_state[:, [0, 1]])
        predict_state_list, _, _, _, _, _, _, _, _, _ = puck_EKF.kalman_filter(init_state, trajectory_tensor[1:],
                                                                               update=False)

        predict_state_list_with_gradient_params = puck_EKF_gradient.kalman_filter(init_state, trajectory_tensor[1:],
                                                                                  update=False)
        predict_state_list_new_coll = puck_EKF_statistik.kalman_filter(init_state, trajectory_tensor[1:], update=False,
                                                                       cal_mode='statistik')
        # with torch.no_grad():
        #     predict_state_list_with_gradient_params_with_res = puck_EKF_gradient.kalman_filter(init_state,
        #                                                                                        trajectory_tensor[1:],
        #                                                                                        update=False, res=res,
        #                                                                                        full_res=full_res)
        predict_state_list = torch.stack(predict_state_list)
        predict_state_list_with_gradient_params = torch.stack(predict_state_list_with_gradient_params[0])
        predict_state_list_new_coll = torch.vstack(predict_state_list_new_coll[0])
        # predict_state_list_with_gradient_params_with_res = torch.stack(predict_state_list_with_gradient_params_with_res[0])
        # plot the table range
        if plot:
            plt.figure()
            plt.subplot(3,2,1)
            plt.title('x')
            plt.scatter(trajectory_tensor[1:, -1], trajectory_tensor[1:, 0], label='record')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list[:,0], label='linear regression')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list_new_coll[:, 0], label='new collision')
            plt.legend()
            plt.subplot(3,2,3)
            plt.title('y')
            plt.scatter(trajectory_tensor[1:, -1], trajectory_tensor[1:, 1], label='record')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list[:, 1], label='linear regression')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list_new_coll[:, 1], label='new collision')
            plt.legend()
            plt.subplot(3,2,5)
            plt.title('theta')
            plt.scatter(trajectory_tensor[1:, -1], trajectory_tensor[1:, 2], label='record')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list[:, 4], label='linear regression')
            plt.scatter(trajectory_tensor[1:, -1], predict_state_list_new_coll[:, 4], label='new collision')
            plt.legend()
            # fig = plt.figure()
            # xy = [0, -1.038 / 2]
            # ax = fig.add_subplot(111)
            # rect = plt.Rectangle(xy, 1.948, 1.038, fill=False)
            # rect.set_linewidth(10)
            # ax.add_patch(rect)
            # s = 60
            # plt.scatter(trajectory_tensor[1, 0], trajectory_tensor[1, 1], s=500, marker='*', c='b')
            # plt.scatter(predict_state_list[:, 0], predict_state_list[:, 1], label='linear regression', s=s, c='r',
            #             alpha=0.5)
            # plt.scatter(predict_state_list_with_gradient_params[:, 0], predict_state_list_with_gradient_params[:, 1],
            #             label='Gradient Descent', s=s, c='y', alpha=0.5)
            # plt.scatter(predict_state_list_new_coll[:, 0], predict_state_list_new_coll[:, 1], label='new collision model',
            #             s=s, c='k', alpha=0.5)
            # plt.scatter(predict_state_list_with_gradient_params_with_res[:, 0],
            #             predict_state_list_with_gradient_params_with_res[:, 1], label='Residual Dynamic', s=s, c='k', alpha=0.5)
            # plt.scatter(trajectory_tensor[1:, 0], trajectory_tensor[1:, 1], label='Real Data', s=s, c='b', alpha=0.5)
            # plt.legend(fontsize=20)
            # plt.xticks([])
            # plt.yticks([])
            # plt.figure()
            # plt.scatter(trajectory_tensor[1:, -1], trajectory_tensor[1:, 2], label='real data', s=s, c='b', alpha=0.5)
            # plt.scatter(trajectory_tensor[1:, -1], predict_state_list_with_gradient_params[:, 4],
            #             label='Gradient Descent', s=s, c='y', alpha=0.5)
            # plt.scatter(trajectory_tensor[1:, -1], predict_state_list[:, 4], label='linear regression', s=s, c='r', alpha=0.5)
            # plt.legend()
    plt.show()
    # x_dynamic_innovation = torch.vstack(dynamic_innovation_list)[:, 0]
    # y_dynamic_innovation = torch.vstack(dynamic_innovation_list)[:, 1]
    # x_observation_innovation = torch.vstack(observation_innovation_list)[:, 0]
    # y_observation_innovation = torch.vstack(observation_innovation_list)[:, 1]
    # print('x dynamic' + str(torch.var(x_dynamic_innovation)) + '\n')
    # print('y dynamic' + str(torch.var(y_dynamic_innovation)) + '\n')
    # print('x observation' + str(torch.var(x_observation_innovation)) + '\n')
    # print('y observation' + str(torch.var(y_observation_innovation)) + '\n')
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.title('x direction dynamic noise')
    # plt.hist(x_dynamic_innovation.cpu().detach().numpy(), density=True, bins=200, range=[-0.005, 0.005])
    # plt.subplot(1, 2, 2)
    # plt.title('y direction dynamic noise')
    # plt.hist(y_dynamic_innovation.cpu().detach().numpy(), density=True, bins=200, range=[-0.005, 0.005])
    # plt.savefig('./dynamic_noise2.jpg')
    # plt.close()
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.title('x direction observation noise')
    # plt.hist(x_observation_innovation.cpu().detach().numpy(), density=True, bins=200, range=[-0.005, 0.005])
    # plt.subplot(1, 2, 2)
    # plt.title('y direction observation noise')
    # plt.hist(y_observation_innovation.cpu().detach().numpy(), density=True, bins=200, range=[-0.005, 0.005])
    # plt.savefig('./observation_noise2.jpg')
    # plt.close

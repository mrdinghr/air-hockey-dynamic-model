from math import pi
import torch
import torch_air_hockey_baseline_no_detach
from matplotlib import pyplot as plt
import numpy as np
from test_params import plot_with_state_list

torch.set_printoptions(threshold=torch.inf)


class AirHockeyEKF:
    def __init__(self, u, system, Q, Q_collision, R, P, device):
        self.state = None
        self.system = system
        self.Q = Q
        self.Q_collision = Q_collision
        self.Qcollision = Q_collision
        self.R = R
        self.P = P
        self.u = u
        self.predict_state = None
        self.F = None
        self.score = False
        self.has_collision = False
        self.H = torch.zeros((3, 6), device=device)
        self.H[0][0] = self.H[1][1] = self.H[2][4] = 1
        self.y = None
        self.S = None
        self.score_time = 0
        self.device = device
        self.pre_collided = False

    def initialize(self, state, P):
        self.state = state
        if P is not None:
            self.P = P
        else:
            self.P = torch.eye(6, device=self.device).float() * 0.01

    def reset_angle(self, state):
        while state[4] > pi:
            state[4] = state[4] - 2 * pi
        while state[4] < -pi:
            state[4] = state[4] + 2 * pi

    # params: table friction, table damping, table restitution, rim friction
    def predict(self, beta=0, res=None, full_res=None, coll_mode=False, cal_mode='spong'):
        transform_matrix = torch.zeros((6, 3), device=self.device)
        transform_matrix[2][0] = transform_matrix[3][1] = transform_matrix[5][2] = 1
        if coll_mode:
            self.has_collision, self.predict_state, jacobian, self.Q_collision, trans_jacobian, collision_mode, slide = self.system.apply_collision(
                self.state, self.Qcollision, beta=beta, coll_mode=coll_mode, cal_mode=cal_mode)
        else:
            self.has_collision, self.predict_state, jacobian, self.Q_collision = self.system.apply_collision(self.state,
                                                                                                             self.Qcollision,
                                                                                                             beta=beta,
                                                                                                             cal_mode=cal_mode)
        if self.has_collision:
            self.F = jacobian.clone()
        else:
            self.F = self.system.F.clone()
            self.predict_state = self.system.f(self.state, self.u, cal_mode=cal_mode)
        if res is not None:
            if self.has_collision:
                self.F = self.F + transform_matrix @ torch.autograd.functional.jacobian(res.cal_res_collision,
                                                                                        self.state,
                                                                                        create_graph=True) * self.u
                self.predict_state = self.predict_state + transform_matrix @ res.cal_res_collision(self.state) * self.u
            else:
                self.F = self.F + transform_matrix @ torch.autograd.functional.jacobian(res.cal_res, self.state,
                                                                                        create_graph=True) * self.u
                self.predict_state = self.predict_state + transform_matrix @ res.cal_res(self.state) * self.u
        if full_res is not None:
            if self.has_collision:
                self.F = self.F + torch.autograd.functional.jacobian(full_res.cal_res_collision, self.state,
                                                                     create_graph=True)
                self.predict_state = self.predict_state + full_res.cal_res_collision(self.state)
            # else:
            #     self.F = self.F + torch.autograd.functional.jacobian(full_res.cal_res, self.state,
            #                                                          create_graph=True)
            #     self.predict_state = self.predict_state + full_res.cal_res(self.state)
        if self.has_collision or self.system.close_boundary(self.state) or self.pre_collided:
            if self.has_collision:
                self.pre_collided = True
            self.P = self.F @ self.P @ self.F.T + self.Q_collision
        else:
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.pre_collided = False
        if coll_mode:
            return trans_jacobian, collision_mode, slide

    def update(self, measure, update=True):
        # measurement residual
        innovation_xy = measure[0:2] - self.predict_state[0:2]
        innovation_theta = measure[2] - self.predict_state[4]
        if innovation_theta >= pi:
            innovation_theta = innovation_theta - pi * 2
        elif innovation_theta <= -pi:
            innovation_theta = innovation_theta + pi * 2
        self.y = torch.cat([innovation_xy, torch.atleast_1d(innovation_theta)])
        if update:
            self.S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ torch.linalg.inv(self.S)
            self.state = self.predict_state + K @ self.y
            self.P = (torch.eye(6, device=self.device) - K @ self.H) @ self.P
        else:
            self.state = self.predict_state
            self.P = self.P

    def refresh(self, P=None, Q=None, R=None, Q_collision=None):
        if P is not None:
            self.P = P
        else:
            self.P = torch.eye(6, device=self.device) * 0.01
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        if Q_collision is not None:
            self.Q_collision = Q_collision
            self.Qcollision = Q_collision

    def smooth(self, init_state, trajectory, plot=False, writer=None, epoch=0, trajectory_index=None,
               cal=None, beta=0, res=None, coll_mode=None, full_res=None, cal_mode='spong'):
        if coll_mode:
            EKF_res_state, EKF_res_collision, innovation_vector, innovation_variance, EKF_res_P, EKF_predict_state, update_list, dynamic_jacobian_list, trans_jacobian_list, coll_mode_list, slide_list, state_for_smooth, variance_for_smooth = self.kalman_filter(
                init_state,
                trajectory,
                beta=beta,
                cal=cal,
                res=res,
                coll_mode=coll_mode)
        else:
            EKF_res_state, EKF_res_collision, innovation_vector, innovation_variance, EKF_res_P, EKF_predict_state, update_list, dynamic_jacobian_list, state_for_smooth, variance_for_smooth = self.kalman_filter(
                init_state,
                trajectory,
                beta=beta,
                cal=cal,
                res=res,
                coll_mode=coll_mode)
        smoothed_state_list, smoothed_variance_list, record_smooth_state, record_smooth_variance = self.backward_pass(
            state_for_smooth,
            variance_for_smooth,
            dynamic_jacobian_list,
            update_list,
            state_for_smooth,
            beta=beta,
            cal=cal, res=res, cal_mode=cal_mode)
        if plot:
            time_list = [i / 120 for i in range(len(EKF_res_state))]
            plot_with_state_list(EKF_res_state, smoothed_state_list, trajectory, time_list, writer=writer, epoch=epoch,
                                 trajectory_index=trajectory_index)
        smoothed_state_list = smoothed_state_list[::-1]
        smoothed_variance_list = smoothed_variance_list[::-1]
        record_smooth_state = record_smooth_state[::-1]
        record_smooth_variance = record_smooth_variance[::-1]
        predict_state_list = self.smooth_one_step_predict(smoothed_state_list)
        update_list = np.array(update_list)
        record_predict_state_list = []
        for i in range(len(predict_state_list)):
            if i in np.where(update_list >= 0)[0]:
                record_predict_state_list.append(predict_state_list[i])
        if coll_mode:
            return record_smooth_state, EKF_res_collision, smoothed_variance_list, record_predict_state_list, trans_jacobian_list, coll_mode_list, slide_list
        return record_smooth_state, record_smooth_variance, record_predict_state_list

    def smooth_one_step_predict(self, smooth_state_list):
        predict_state_list = [smooth_state_list[0]]
        for i in range(len(smooth_state_list) - 1):
            self.state = smooth_state_list[i]
            self.predict()
            predict_state_list.append(self.predict_state)
        return predict_state_list

    def backward_pass(self, state_list, variance_list, jacobian_list, update_list, full_time_state_list, beta=0,
                      cal=None, res=None, cal_mode='spong'):
        smoothed_state_list = [state_list[-1]]
        if update_list[-1] >= 0:
            record_smooth_state_list = [state_list[-1]]
        else:
            record_smooth_state_list = []
        smoothed_variance_list = [self.H @ variance_list[-1] @ self.H.T + self.R]
        record_smooth_variance_list = [self.H @ variance_list[-1] @ self.H.T + self.R]
        xs = smoothed_state_list[-1]
        ps = variance_list[-1]
        time = len(full_time_state_list)
        for j in range(time - 1):
            idx_prev = - j - 2
            if cal is not None:
                params = cal.cal_params(full_time_state_list[idx_prev])
                self.system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                                       tableFrictionY=params[3], restitution=params[4],
                                       rimFriction=params[5])
            has_collision, predict_state, _, self.Q_collision = self.system.apply_collision(
                full_time_state_list[idx_prev], beta=beta, Q_collision=self.Qcollision, cal_mode=cal_mode)
            if not has_collision:
                xp = self.system.f(full_time_state_list[idx_prev], self.u, cal_mode=cal_mode)
            else:
                xp = predict_state
            if xs[4] - xp[4] > 3 / 2 * pi:
                xp4 = xp[4] + 2 * pi
            elif xs[4] - xp[4] < -3 / 2 * pi:
                xp4 = xp[4] - 2 * pi
            else:
                xp4 = xp[4]
            if res is not None:
                res_state = res.cal_res(full_time_state_list[idx_prev])
                xp_new = torch.cat(
                    [xp[0:4] + res_state[0:4], torch.atleast_1d(xp4 + res_state[4]),
                     torch.atleast_1d(xp[5] + res_state[5])])
            else:
                xp_new = torch.cat([xp[0:4], torch.atleast_1d(xp4), torch.atleast_1d(xp[5])])
            if has_collision:
                predicted_cov = jacobian_list[idx_prev] @ variance_list[idx_prev] @ jacobian_list[
                    idx_prev].T + self.Q_collision
            else:
                predicted_cov = jacobian_list[idx_prev] @ variance_list[idx_prev] @ jacobian_list[idx_prev].T + self.Q
            smooth_gain = variance_list[idx_prev] @ jacobian_list[idx_prev].T @ torch.linalg.inv(predicted_cov)
            if update_list[idx_prev + 1] > 0:
                xs = full_time_state_list[idx_prev] + smooth_gain @ (xs - xp_new)
                ps = variance_list[idx_prev] + smooth_gain @ (ps - predicted_cov) @ smooth_gain.T
            else:
                xs = full_time_state_list[idx_prev]
                ps = variance_list[idx_prev]
            if update_list[idx_prev + 1] >= 0:
                record_smooth_state_list.append(xs)
                record_smooth_variance_list.append(self.H @ ps @ self.H.T + self.R)
            smoothed_state_list.append(xs)
            smoothed_variance_list.append(self.H @ ps @ self.H.T + self.R)
        return smoothed_state_list, smoothed_variance_list, record_smooth_state_list, record_smooth_variance_list

    def kalman_filter(self, init_state, trajectory, cal=None, beta=0, update=True, res=None, P=None, full_res=None,
                      coll_mode=False, cal_mode='spong'):
        self.initialize(init_state, P=P)
        EKF_res_state = [init_state.clone()]
        state_for_smooth = [init_state.clone()]
        EKF_res_P = [self.P]
        variance_for_smooth = [self.P]
        innovation_vector = [torch.zeros(3, device=self.device)]
        innovation_variance = [self.H @ self.P @ self.H.T + self.R]
        EKF_res_collision = [False]
        EKF_predict_state = [init_state.clone()]
        i = 0
        j = 0
        length = len(trajectory)
        time_EKF = [0]
        trans_jacobian_list = [torch.eye(6, device=self.device)]
        coll_mode_list = ['']
        slide_list = [0]
        update_list = [1]
        dynamic_jacobian_list = []
        while j < length - 1:
            i += 1
            time_EKF.append(i / 120)
            if cal is not None:
                params = cal.cal_params(self.state)
                self.system.set_params(tableDampingX=params[0], tableDampingY=params[1], tableFrictionX=params[2],
                                       tableFrictionY=params[3], restitution=params[4],
                                       rimFriction=params[5])
            if coll_mode:
                trans_jacobian, collision_mode, slide = self.predict(beta=beta, res=res, full_res=full_res,
                                                                     coll_mode=True, cal_mode=cal_mode)
            else:
                self.predict(beta=beta, res=res, full_res=full_res, cal_mode=cal_mode)
            if (i - 0.5) / 120 <= trajectory[j + 1][-1] - trajectory[0][-1] <= (i + 0.5) / 120:
                EKF_predict_state.append(self.predict_state)
                self.update(trajectory[j + 1][0:3], update=update)
                j += 1
                EKF_res_state.append(self.state)
                EKF_res_P.append(self.P)
                innovation_vector.append(self.y)
                innovation_variance.append(self.S)
                EKF_res_collision.append(self.has_collision)
                update_list.append(1)
                dynamic_jacobian_list.append(self.F)
                if coll_mode:
                    trans_jacobian_list.append(trans_jacobian)
                    coll_mode_list.append(collision_mode)
                    slide_list.append(slide)
            elif trajectory[j + 1][-1] - trajectory[0][-1] < (i - 0.5) / 120:
                j = j + 1
                i = i - 1
                EKF_predict_state.append(self.predict_state)
                self.state = self.predict_state
                EKF_res_P.append(self.P)
                EKF_res_state.append(self.state)
                EKF_res_collision.append(self.has_collision)
                update_list.append(0)
                dynamic_jacobian_list.append(self.F)
                if coll_mode:
                    trans_jacobian_list.append(trans_jacobian)
                    coll_mode_list.append(collision_mode)
                    slide_list.append(slide)
            else:
                self.state = self.predict_state
                update_list.append(-1)
                dynamic_jacobian_list.append(self.F)
            state_for_smooth.append(self.state)
            variance_for_smooth.append(self.P)
        dynamic_jacobian_list.append(self.F)
        if coll_mode:
            return EKF_res_state, EKF_res_collision, innovation_vector, innovation_variance, EKF_res_P, EKF_predict_state, update_list, dynamic_jacobian_list, trans_jacobian_list, coll_mode_list, slide_list, state_for_smooth, variance_for_smooth
        return EKF_res_state, EKF_res_collision, innovation_vector, innovation_variance, EKF_res_P, EKF_predict_state, update_list, dynamic_jacobian_list, state_for_smooth, variance_for_smooth


if __name__ == '__main__':
    device = torch.device("cuda")
    # test for torch_EKF_Wrapper
    # tableDamping = 0.001
    # tableFriction = 0.001
    # tableRestitution = 0.7424
    para = [0.10608561, 0.34085548, 0.78550678]
    system = torch_air_hockey_baseline_no_detach.SystemModel(tableDamping=para[1], tableFriction=para[0],
                                                             tableLength=1.948,
                                                             tableWidth=1.038,
                                                             goalWidth=0.25, puckRadius=0.03165, malletRadius=0.04815,
                                                             tableRes=para[2], malletRes=0.8, rimFriction=0.1418,
                                                             dt=1 / 120, beta=30)
    R = torch.zeros((3, 3), device=device)
    R[0][0] = 2.5e-7
    R[1][1] = 2.5e-7
    R[2][2] = 9.1e-3
    Q = torch.zeros((6, 6), device=device)
    Q[0][0] = Q[1][1] = 2e-10
    Q[2][2] = Q[3][3] = 1e-7
    Q[4][4] = 1.0e-2
    Q[5][5] = 1e-1
    P = torch.eye(6, device=device) * 0.01
    pre_data = np.load("total_data.npy", allow_pickle=True)
    pre_data = pre_data[0]
    data = []
    for i in range(1, len(pre_data)):
        if abs(pre_data[i][0] - pre_data[i - 1][0]) < 0.005 and abs(pre_data[i][1] - pre_data[i - 1][1]) < 0.005:
            continue
        data.append(pre_data[i])
    for i_data in data:
        i_data[0] += system.table.m_length / 2
    data = torch.tensor(np.array(data), device=device).float()
    state_dx = ((data[1][0] - data[0][0]) / (data[1][3] - data[0][3]) + (
            data[2][0] - data[1][0]) / (
                        data[2][3] - data[1][3]) + (data[3][0] - data[2][0]) / (
                        data[3][3] - data[2][3])) / 3
    state_dy = ((data[1][1] - data[0][1]) / (data[1][3] - data[0][3]) + (
            data[2][1] - data[1][1]) / (
                        data[2][3] - data[1][3]) + (data[3][1] - data[2][1]) / (
                        data[3][3] - data[2][3])) / 3
    state_dtheta = ((data[1][2] - data[0][2]) / (data[1][3] - data[0][3]) + (
            data[2][2] - data[1][2]) / (
                            data[2][3] - data[1][3]) + (data[3][2] - data[2][2]) / (
                            data[3][3] - data[2][3])) / 3
    state = torch.tensor([data[1][0], data[1][1], state_dx, state_dy, data[1][2], state_dtheta], device=device)
    puck_EKF = AirHockeyEKF(u=1 / 120, system=system, Q=Q, R=R, P=P)
    puck_EKF.initialize(state)
    resx = [state[0]]
    resy = [state[1]]
    res_theta = [state[4]]
    time_EKF = [1 / 120]
    j = 1
    length = len(data) - 1
    i = 0
    evaluation = 0
    num_evaluation = 0
    while j < length:
        i += 1
        time_EKF.append((i + 1) / 120)
        puck_EKF.predict()
        resx.append(puck_EKF.predict_state[0])
        resy.append(puck_EKF.predict_state[1])
        res_theta.append(puck_EKF.predict_state[4])
        # check whether data is recorded at right time
        if (i - 0.2) / 120 < data[j + 1][-1] - data[1][-1] < (i + 0.2) / 120:
            puck_EKF.update(data[j + 1][0:3])
            j += 1
            sign, logdet = torch.linalg.slogdet(puck_EKF.S)
            num_evaluation += 1
            evaluation += sign * torch.exp(logdet) + puck_EKF.y @ torch.linalg.inv(puck_EKF.S) @ puck_EKF.y
        elif data[j + 1][-1] - data[1][-1] <= (i - 0.2) / 120:
            j += 1
            puck_EKF.state = puck_EKF.predict_state
        else:
            puck_EKF.state = puck_EKF.predict_state
    print(evaluation / num_evaluation)
    resx = torch.tensor(resx, device=device)
    resy = torch.tensor(resy, device=device)
    res_theta = torch.tensor(res_theta, device=device)
    data_x_velocity = []
    data_y_velocity = []
    data_theta_velocity = []
    for i in range(1, len(pre_data)):
        data_x_velocity.append((pre_data[i][0] - pre_data[i - 1][0]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
        data_y_velocity.append((pre_data[i][1] - pre_data[i - 1][1]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
        if abs(pre_data[i][2] - pre_data[i - 1][2]) > pi:
            data_theta_velocity.append(
                (pre_data[i][2] - np.sign(pre_data[i][2]) * pi) / (pre_data[i][-1] - pre_data[i - 1][-1]))
    else:
        data_theta_velocity.append((pre_data[i][2] - pre_data[i - 1][2]) / (pre_data[i][-1] - pre_data[i - 1][-1]))
    plt.figure()
    plt.scatter(data[1:, 0].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g', label='raw data', s=5)
    plt.scatter(resx.cpu().numpy(), resy.cpu().numpy(), color='b', label='EKF', s=5)
    plt.legend()
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.scatter(time_EKF, resx.cpu().numpy(), color='b', label='EKF x position', s=5)
    plt.title('only EKF x position')
    plt.legend()
    plt.subplot(3, 3, 2)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g',
                label='raw data x position', s=5)
    plt.title('only raw data x position')
    plt.legend()
    plt.subplot(3, 3, 3)
    plt.scatter(time_EKF, resx.cpu().numpy(), color='b', label='EKF x position', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 0].cpu().numpy(), color='g',
                label='raw data x position', s=5)
    plt.title('EKF vs raw data x position')
    plt.legend()
    plt.subplot(3, 3, 4)
    plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
    plt.title('only EKF y position')
    plt.legend()
    plt.subplot(3, 3, 5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.title('only raw data y position')
    plt.legend()
    plt.subplot(3, 3, 6)
    plt.scatter(time_EKF, resy.cpu().numpy(), color='b', label='EKF y position', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 1].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.title('EKF vs raw data y position')
    plt.legend()
    plt.subplot(3, 3, 7)
    plt.scatter(time_EKF, res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
    plt.title('only EKF theta')
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g',
                label='raw data theta', s=5)
    plt.legend()
    plt.subplot(3, 3, 9)
    plt.scatter(time_EKF, res_theta.cpu().numpy(), color='b', label='EKF theta', s=5)
    plt.scatter(data[1:, -1].cpu().numpy() - data[0][-1].cpu().numpy(), data[1:, 2].cpu().numpy(), color='g',
                label='raw data y position', s=5)
    plt.legend()
    plt.show()

import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch_air_hockey_baseline_no_detach as torch_air_hockey_baseline
from torch_EKF_Wrapper import AirHockeyEKF
from math import pi
from test_params import EKF_plot_with_state_list
from tqdm import tqdm

torch.set_printoptions(threshold=torch.inf)


def judge_time_alignment(trajectory):
    for j in range(1, len(trajectory)):
        if not ((j - 0.5) / 120 <= trajectory[j, -1] - trajectory[0, -1] <= (j + 0.5) / 120):
            return False
    return True


class Kalman_EKF_Gradient(torch.nn.Module):
    def __init__(self, params, covariance_params, covariance_params_collision, segment_size, device, cal=None):
        super(Kalman_EKF_Gradient, self).__init__()
        if cal is not None:
            self.register_parameter('params', torch.nn.Parameter(params))
        else:
            self.params = params
        # self.register_parameter('covariance_params', torch.nn.Parameter(covariance_params))
        self.covariance_params = covariance_params
        self.covariance_params_collision = covariance_params_collision
        self.segment_size = segment_size
        self.device = device
        self.dyna_params = None
        # self.covariance_params = covariance_params
        self.puck_EKF = None

    def construct_EKF(self):
        self.dyna_params = torch.abs(self.params)
        R = torch.diag(torch.stack([self.covariance_params[0],
                                    self.covariance_params[1],
                                    self.covariance_params[2]]))
        Q = torch.diag(torch.stack([self.covariance_params[3], self.covariance_params[3],
                                    self.covariance_params[4], self.covariance_params[4],
                                    self.covariance_params[5], self.covariance_params[6]]))
        Q_collision = torch.diag(
            torch.stack([self.covariance_params_collision[3], self.covariance_params_collision[3],
                         self.covariance_params_collision[4], self.covariance_params_collision[4],
                         self.covariance_params_collision[5], self.covariance_params_collision[6]]))
        P = torch.eye(6, device=self.device) * 0.01
        dynamic_system = torch_air_hockey_baseline.SystemModel(tableDampingX=self.dyna_params[0],
                                                               tableDampingY=self.dyna_params[1],
                                                               tableFrictionX=self.dyna_params[2],
                                                               tableFrictionY=self.dyna_params[3],
                                                               tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                                               puckRadius=0.03165, malletRadius=0.04815,
                                                               tableRes=self.dyna_params[4],
                                                               malletRes=0.8, rimFriction=self.dyna_params[5],
                                                               dt=1 / 120, device=self.device)
        self.puck_EKF = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=self.device,
                                     Q_collision=Q_collision)

    def prepare_dataset(self, trajectory_buffer, writer=None, plot=False, epoch=0, type='EKF', cal=None, beta=0,
                        update=True, res=None, tag='', save_gap=5, full_res=None):
        if type == 'EKF':
            self.construct_EKF()
            segments_dataset = []
            with torch.no_grad():
                for trajectory_index, trajectory in enumerate(trajectory_buffer):
                    trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                    init_state = self.calculate_init_state(trajectory)
                    EKF_state, collisions, _, _, _, _, _, _, _, _ = self.puck_EKF.kalman_filter(init_state, trajectory_tensor[1:],
                                                                                    cal=cal, beta=beta, update=update,
                                                                                    res=res, full_res=full_res)
                    if plot:
                        predict_state, _, _, _, _, _, _, _, _, _ = self.puck_EKF.kalman_filter(init_state, trajectory_tensor[1:],
                                                                                   cal=cal, beta=beta, update=False,
                                                                                   res=res, full_res=full_res)
                        EKF_plot_with_state_list(EKF_state_list=EKF_state, trajectory=trajectory_tensor[1:],
                                                 prediction=predict_state, writer=writer, epoch=epoch,
                                                 trajectory_index=trajectory_index, tag=tag, save_gap=save_gap)
                    segment = self.construct_data_segments(EKF_state, collisions, trajectory_index,
                                                           trajectory_tensor[1:])
                    if segment != []:
                        segments_dataset.append(torch.vstack(segment))
            return torch.vstack(segments_dataset)
        if type == 'smooth':
            self.construct_EKF()
            segments_dataset = []
            # Pure Kalman Smooth
            with torch.no_grad():
                for trajectory_index, trajectory in enumerate(trajectory_buffer):
                    trajectory_tensor = torch.tensor(trajectory, device=self.device).float()
                    init_state = self.calculate_init_state(trajectory)
                    smoothed_states, smoothed_variances, collisions = self.puck_EKF.smooth(init_state,
                                                                                           trajectory_tensor[1:],
                                                                                           plot=plot, writer=writer,
                                                                                           epoch=epoch,
                                                                                           trajectory_index=trajectory_index,
                                                                                           cal=cal, beta=beta,
                                                                                           res=res)
                    segments_dataset.append(
                        torch.vstack(
                            self.construct_data_segments(smoothed_states, collisions, trajectory_index)))
            return torch.vstack(segments_dataset)

    def calculate_init_state(self, trajectory):
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
                              device=self.device).float()
        return state_



    def construct_data_segments(self, EKF_state, collisions, trajectory_index, recorded_trajectory):
        segment_dataset = []
        num_collision = 0
        num_no_collision = 0
        for j in range(1, len(EKF_state) - self.segment_size + 1):
            if not judge_time_alignment(recorded_trajectory[j:j+self.segment_size]):
                continue
            if np.any(collisions[j + 1: j + self.segment_size]):
                if np.any(collisions[j:j + 3]):
                    continue
                cur_state = EKF_state[j]
                index_tensor = torch.tensor([trajectory_index, j + 1], device=self.device)
                segment_dataset.append(torch.cat((index_tensor, cur_state)))
                num_collision += 1
            else:
                if j % (self.segment_size) == 0:
                    if np.any(collisions[j:j + 3]):
                        continue
                    cur_state = EKF_state[j]
                    index_tensor = torch.tensor([trajectory_index, j + 1], device=self.device)
                    segment_dataset.append(torch.cat((index_tensor, cur_state)))
                    num_no_collision += 1
        # print('collision: ', num_collision, 'no collision', num_no_collision)
        return segment_dataset

    def calculate_loss(self, segments_batch, measurements, loss_type='log_like', type='EKF', epoch=0,
                       cal=None, beta=0, res=None, full_res=None):
        if type == 'EKF':
            self.construct_EKF()
            total_loss = 0
            num_total_loss = 0
            for point in segments_batch:
                trajectory_idx = int(point[0])
                trajectory_point_idx = int(point[1])
                segment_measurment = torch.tensor(
                    measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size],
                    device=self.device).float()
                EKF_state, _, innovation_vectors, innovation_variance, res_P, EKF_predict_state, _, _, _, _ = self.puck_EKF.kalman_filter(
                    point[2:], segment_measurment, cal=cal, update=True, beta=beta, res=res, full_res=full_res)
                if loss_type == 'multi_log_like' or loss_type == 'multi_mse':
                    predict_state_list, _, innovation_vectors, innovation_variance, EKF_res_P, _, _, _, _, _ = self.puck_EKF.kalman_filter(
                        point[2:], segment_measurment, cal=cal, update=False, beta=beta, res=res, full_res=full_res)
                    EKF_variance_tensor = torch.stack(innovation_variance[1:])
                    innovation_vectors = torch.stack(innovation_vectors)
                elif loss_type == 'xyomega_log_like' or loss_type == 'xyomega_mse':
                    EKF_variance = []
                    for i in range(1, len(res_P)):
                        cur_variance = torch.zeros((3, 3), device=self.device)
                        cur_variance[0:2] = res_P[i][0:2, [0, 1, 5]]
                        cur_variance[2] = res_P[i][5, [0, 1, 5]]
                        EKF_variance.append(cur_variance)
                    EKF_variance_tensor = torch.stack(EKF_variance)
                    innovation = torch.stack(EKF_predict_state)[1:, [0, 1, 5]] - torch.stack(EKF_state)[1:, [0, 1, 5]]
                elif loss_type == 'vomega_log_like' or loss_type == 'vomega_mse':
                    EKF_variance = []
                    for i in range(1, len(res_P)):
                        cur_variance = torch.zeros((3, 3), device=self.device)
                        cur_variance[0:2] = res_P[i][2:4, [2, 3, 5]]
                        cur_variance[2] = res_P[i][5, [2, 3, 5]]
                        EKF_variance.append(cur_variance)
                    EKF_variance_tensor = torch.stack(EKF_variance)
                    innovation = torch.stack(EKF_predict_state)[1:, [2, 3, 5]] - torch.stack(EKF_state)[1:, [2, 3, 5]]
                elif loss_type == 'full_log_like' or loss_type == 'full_mse':
                    EKF_variance = []
                    for i in range(1, len(res_P)):
                        cur_variance = res_P[i]
                        EKF_variance.append(cur_variance)
                    EKF_variance_tensor = torch.stack(EKF_variance)
                    innovation_vectors = torch.stack(EKF_predict_state) - torch.stack(EKF_state)
                    innovation_vectors2 = innovation_vectors[1:, 4]
                    idx = torch.where(innovation_vectors[1:, 4] > 3 / 2 * np.pi)[0]
                    innovation_vectors2[idx] = innovation_vectors2[idx] - 2 * np.pi
                    idx = torch.where(innovation_vectors[1:, 4] < -3 / 2 * np.pi)[0]
                    innovation_vectors2[idx] = innovation_vectors2[idx] + 2 * np.pi
                    innovation = torch.cat([innovation_vectors[1:, 0:4], innovation_vectors2.unsqueeze(1),
                                            innovation_vectors[1:, 5].unsqueeze(1)], dim=1)
                elif loss_type == 'mse' or loss_type == 'log_like':
                    EKF_variance_tensor = torch.stack(innovation_variance[1:])
                    innovation_vectors = torch.stack(innovation_vectors)
                sign, logdet = torch.linalg.slogdet(EKF_variance_tensor)
                if loss_type in ['multi_mse', 'multi_log_like', 'log_like', 'mse']:
                    innovation_vectors2 = innovation_vectors[1:, 2]
                    idx = torch.where(innovation_vectors[1:, 2] > 3 / 2 * np.pi)[0]
                    innovation_vectors2[idx] = innovation_vectors2[idx] - 2 * np.pi
                    idx = torch.where(innovation_vectors[1:, 2] < -3 / 2 * np.pi)[0]
                    innovation_vectors2[idx] = innovation_vectors2[idx] + 2 * np.pi
                    innovation = torch.cat([innovation_vectors[1:, 0:2], innovation_vectors2.unsqueeze(1)], dim=1)
                if loss_type == 'log_like' or loss_type == 'multi_log_like' or loss_type == 'xyomega_log_like' or loss_type == 'vomega_log_like' or loss_type == 'full_log_like':
                    loss_i = torch.einsum('ij, ijk, ik->i', innovation, torch.linalg.inv(EKF_variance_tensor),
                                          innovation)
                    total_loss = total_loss + 0.5 * torch.sum(logdet + loss_i)
                elif loss_type == 'mse' or loss_type == 'multi_mse' or loss_type == 'xyomega_mse' or loss_type == 'vomega_mse' or loss_type == 'full_mse':
                    total_loss = total_loss + torch.bmm(innovation.unsqueeze(1), innovation.unsqueeze(2)).sum()
                num_total_loss += 1
            return total_loss / num_total_loss
        if type == 'smooth':
            self.construct_EKF()
            total_loss = 0
            num_total_loss = 0
            for point in segments_batch:
                trajectory_idx = int(point[0])
                trajectory_point_idx = int(point[1])
                segment_measurment = torch.tensor(
                    measurements[trajectory_idx][trajectory_point_idx:trajectory_point_idx + self.segment_size],
                    device=self.device).float()
                smoothed_state_list, smoothed_variance_list, _ = self.puck_EKF.smooth(point[2:], segment_measurment,
                                                                                      epoch=epoch, beta=beta,
                                                                                      cal=cal, res=res)
                smoothed_state_tensor = torch.stack(smoothed_state_list[1:])
                smoothed_variance_tensor = torch.stack(smoothed_variance_list[1:])

                innovation_xy = segment_measurment[1:, :2] - smoothed_state_tensor[:, :2]
                innovation_angle = segment_measurment[1:, 2] - smoothed_state_tensor[:, 4]
                sign, logdet = torch.linalg.slogdet(smoothed_variance_tensor)
                idx = torch.where(segment_measurment[1:, 2] - smoothed_state_tensor[:, 4] > 3 / 2 * np.pi)[0]
                innovation_angle[idx] = innovation_angle[idx] - 2 * np.pi

                idx = torch.where(segment_measurment[1:, 2] - smoothed_state_tensor[:, 4] < -3 / 2 * np.pi)[0]
                innovation_angle[idx] = innovation_angle[idx] + 2 * np.pi

                innovation = torch.cat([innovation_xy, innovation_angle.unsqueeze(1)], dim=1)
                if loss_type == 'log_like':
                    loss_i = torch.einsum('ij, ijk, ik->i', innovation, torch.linalg.inv(smoothed_variance_tensor),
                                          innovation)
                    total_loss = total_loss + 0.5 * torch.sum(logdet + loss_i)
                elif loss_type == 'mse':
                    total_loss = total_loss + torch.bmm(innovation.unsqueeze(1), innovation.unsqueeze(2)).sum()
                num_total_loss += 1

            return total_loss / num_total_loss


def load_dataset(file_name):
    total_dataset = np.load(file_name, allow_pickle=True)
    np.random.shuffle(total_dataset)
    return total_dataset[0:(len(total_dataset) - 20)], total_dataset[(len(total_dataset) - 20):]


if __name__ == '__main__':
    device = torch.device("cuda")
    file_name = 'new_total_data_no_collision.npy'
    torch.manual_seed(0)
    lr = 1e-4
    batch_size = 2
    batch_trajectory_size = 10
    epochs = 1000
    training_dataset, test_dataset = load_dataset(file_name)

    # table friction, table damping, table restitution, rim friction
    init_params = torch.Tensor([0.5, 0.5, 0.5, 0.5]).to(device=device)
    # init_params = init_params / torch.tensor([0.1, 0.1, 1, 1]) - 1
    # init_params = 0.5 * (torch.log(1 + init_params) - torch.log(1 - init_params))
    #                                      R0, R1, R2, Q01, Q23, Q4, Q5
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device)
    # covariance_params = torch.Tensor(
    #     [0.00118112, 0.00100000, 0.00295336, 0.00392161, 0.00100000, 0.00100000, 0.00205159]).to(device=device)
    covariance_params = torch.log(covariance_params)
    # covariance_params = torch.log(covariance_params / (1-covariance_params))
    model = Kalman_EKF_Gradient(init_params, covariance_params, segment_size=batch_trajectory_size, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    writer = SummaryWriter('./alldata/718nn' + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M-%S"))
    for t in tqdm(range(epochs)):
        writer.add_scalar('dynamics/table damping', model.params[1], t)
        writer.add_scalar('dynamics/table friction', model.params[0], t)
        writer.add_scalar('dynamics/table restitution', model.params[2], t)
        writer.add_scalar('dynamics/rim friction', model.params[3], t)
        writer.add_scalar('covariance/R0', torch.exp(model.covariance_params[0]), t)
        writer.add_scalar('covariance/R1', torch.exp(model.covariance_params[1]), t)
        writer.add_scalar('covariance/R2', torch.exp(model.covariance_params[2]), t)
        writer.add_scalar('covariance/Q01', torch.exp(model.covariance_params[3]), t)
        writer.add_scalar('covariance/Q23', torch.exp(model.covariance_params[4]), t)
        writer.add_scalar('covariance/Q4', torch.exp(model.covariance_params[5]), t)
        writer.add_scalar('covariance/Q5', torch.exp(model.covariance_params[6]), t)
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=True,
                                                         type='smooth')
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type='smooth', epoch=t)
            if loss.requires_grad:
                loss.backward()
                print("loss:", loss.item())
                print("dynamics: ", model.params.detach().cpu().numpy())
                optimizer.step()
                batch_loss.append(loss.detach().cpu().numpy())
            else:
                print("loss:", loss.item())
                print("dynamics: ", model.params.detach().cpu().numpy())
                batch_loss.append(loss.cpu().numpy())
            # loss.backward()
            # print("loss:", loss.item())
            # print("dynamics: ", model.params.detach().cpu().numpy())
            # optimizer.step()
            # batch_loss.append(loss.detach().cpu().numpy())
        training_loss = np.mean(batch_loss)
        writer.add_scalar('loss/training_loss', training_loss, t)
        with torch.no_grad():
            plot_trajectory(abs(model.params), training_dataset, epoch=t, writer=writer)
        test_segment_dataset = model.prepare_dataset(test_dataset, type='smooth', epoch=t)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type='smooth', epoch=t)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)
    writer.close()

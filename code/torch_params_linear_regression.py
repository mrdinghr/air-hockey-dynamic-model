from torch_air_hockey_baseline_no_detach import SystemModel
from torch_EKF_Wrapper import AirHockeyEKF
import torch
import numpy as np
from total_test_plot_other_use import calculate_init_state
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_gradient import judge_time_alignment
from cov_dyna_params import generate_params
import matplotlib.pyplot as plt


# Linear Regression calculate dynamics parameters
# needed: input filter state, output filter state and dynamics type:non_collision or which mode collision
def linear_regression_dynamics_params(filter_state_list, collision_mode_list, trajectory_set, transform_jacobian_list,
                                      slide_list, device=torch.device('cpu')):
    j = 0
    total_length = 0
    # divide collision and 2 mode non-collision part
    non_coll_input = []
    coll_slide_input = []
    coll_non_slide_input = []
    non_coll_output = []
    coll_slide_output = []
    coll_non_slide_output = []
    for i in range(len(filter_state_list)):
        if i == total_length:
            if collision_mode_list[i + 1] == '':
                non_coll_input.append(filter_state_list[i])
            elif collision_mode_list[i + 1] == 'slide':
                coll_slide_input.append(transform_jacobian_list[i + 1] @ filter_state_list[i])
            else:
                coll_non_slide_input.append(transform_jacobian_list[i + 1] @ filter_state_list[i])
        elif i == total_length + len(trajectory_set[j]) - 2:
            total_length = i + 1
            j += 1
            if collision_mode_list[i] == '':
                non_coll_output.append(filter_state_list[i])
            elif collision_mode_list[i] == 'slide':
                coll_slide_output.append(transform_jacobian_list[i] @ filter_state_list[i])
            else:
                coll_non_slide_output.append(transform_jacobian_list[i] @ filter_state_list[i])
        else:
            if i < len(filter_state_list) - 1:
                if collision_mode_list[i + 1] == '':
                    non_coll_input.append(filter_state_list[i])
                elif collision_mode_list[i + 1] == 'slide':
                    coll_slide_input.append(transform_jacobian_list[i + 1] @ filter_state_list[i])
                else:
                    coll_non_slide_input.append(transform_jacobian_list[i + 1] @ filter_state_list[i])
            if collision_mode_list[i] == '':
                non_coll_output.append(filter_state_list[i])
            elif collision_mode_list[i] == 'slide':
                coll_slide_output.append(transform_jacobian_list[i] @ filter_state_list[i])
            else:
                coll_non_slide_output.append(transform_jacobian_list[i] @ filter_state_list[i])
    non_coll_input = torch.vstack(non_coll_input)
    if coll_slide_input != [] or coll_non_slide_input != []:
        coll_input = torch.vstack(coll_non_slide_input + coll_slide_input)
        coll_output = torch.vstack(coll_non_slide_output + coll_slide_output)
    if coll_slide_input != []:
        coll_slide_input = torch.vstack(coll_slide_input)
        coll_slide_output = torch.vstack(coll_slide_output)
    non_coll_output = torch.vstack(non_coll_output)
    # use correspond input and output with linear regression method to calculate new dynamic parameters
    new_table_dampingx = 120 * non_coll_input[:, 2] @ (non_coll_input[:, 2] - non_coll_output[:, 2]) / (
            non_coll_input[:, 2] @ non_coll_input[:, 2])
    new_table_dampingy = 120 * non_coll_input[:, 3] @ (non_coll_input[:, 3] - non_coll_output[:, 3]) / (
            non_coll_input[:, 3] @ non_coll_input[:, 3])
    # new_table_dampingx = puck_EKF.system.table.tableDampingX
    # new_table_dampingy = puck_EKF.system.table.tableDampingY
    if coll_slide_input != [] or coll_non_slide_input != []:
        new_table_restitution = -coll_input[:, 3] @ coll_output[:, 3] / (coll_input[:, 3] @ coll_input[:, 3])
    else:
        new_table_restitution = puck_EKF.system.table.m_e
    slide_list = torch.tensor(slide_list, device=device)
    if coll_slide_input != []:
        b_diff = (coll_slide_output[:, 2] - coll_slide_input[:, 2]) * slide_list[torch.where(slide_list != 0)]
        theta_diff = (coll_slide_output[:, 5] - coll_slide_input[:, 5]) * slide_list[
            torch.where(slide_list != 0)] * 0.03165 / 2
        n_input = torch.cat([coll_slide_input[:, 3], coll_slide_input[:, 3]])
        new_table_rimfriction = torch.cat([b_diff, theta_diff]) @ n_input / (
                (n_input @ n_input) * (1 + puck_EKF.system.table.m_e))
    else:
        new_table_rimfriction = puck_EKF.system.table.m_rimFriction
    return [new_table_dampingx, new_table_dampingy, new_table_restitution, new_table_rimfriction]


# calculate new R Q and Q_collision
# needed: filter state / predict state and recorded state with correspond dynamics type
def calculate_variance(filter_state_list, predict_state_list, trajectory_set, collision_mode_list, trans_jac_list):
    filter_state_list = torch.vstack(filter_state_list)
    predict_state_list = torch.vstack(predict_state_list)
    collision_mode_list = np.array(collision_mode_list)
    trans_jac_list = torch.stack(trans_jac_list)
    observation_innovation = filter_state_list[:, [0, 1, 4]] - trajectory_set[:, 0:3]
    observation_diag = [torch.var(observation_innovation[:, 0]), torch.var(observation_innovation[:, 1]),
                        torch.var(observation_innovation[:, 2])]
    new_R = torch.diag(torch.stack(observation_diag))
    collision_index = np.where(collision_mode_list != '')[0]
    non_collision_index = np.where(collision_mode_list == '')[0]
    dynamic_innovation = filter_state_list - predict_state_list
    if not np.any(collision_index):
        new_Q_collision = puck_EKF.Q_collision
    else:
        dynamic_collision_innovation = torch.einsum('ikj, ij->ik', trans_jac_list[collision_index],
                                                    dynamic_innovation[collision_index])
        dynamic_collision_diag = [torch.var(dynamic_collision_innovation[:, 0]),
                                  torch.var(dynamic_collision_innovation[:, 1]),
                                  torch.var(dynamic_collision_innovation[:, 2]),
                                  torch.var(dynamic_collision_innovation[:, 3]),
                                  torch.var(dynamic_collision_innovation[:, 4]),
                                  torch.var(dynamic_collision_innovation[:, 5])]
        new_Q_collision = torch.diag(torch.stack(dynamic_collision_diag))
    dynamic_non_collision_innovation = torch.einsum('ikj, ij->ik', trans_jac_list[non_collision_index],
                                                    dynamic_innovation[non_collision_index])
    dynamic_non_collision_diag = [torch.var(dynamic_non_collision_innovation[:, 0]),
                                  torch.var(dynamic_non_collision_innovation[:, 1]),
                                  torch.var(dynamic_non_collision_innovation[:, 2]),
                                  torch.var(dynamic_non_collision_innovation[:, 3]),
                                  torch.var(dynamic_non_collision_innovation[:, 4]),
                                  torch.var(dynamic_non_collision_innovation[:, 5])]
    new_Q = torch.diag(torch.stack(dynamic_non_collision_diag))
    return new_R, new_Q, new_Q_collision


data_file_name = 'hundred_data_one_coll.npy'
save_path = './alldata/linear_regression/smooth/non_collision1123'
filter_type = 'smooth' # smooth EKF
save = False
# total_dataset = np.load(data_file_name, allow_pickle=True)
dataset = np.load(data_file_name, allow_pickle=True)
total_dataset = []
for data in dataset:
    if not judge_time_alignment(data):
        continue
    total_dataset.append(data)
total_dataset = np.array(total_dataset)
device = torch.device("cpu")
epoch = 100
P, R, Q, Q_collision, dyna_params, dynamic_system = generate_params(device)
R = torch.eye(3, device=device)
Q = torch.eye(6, device=device)
Q_collision = torch.eye(6, device=device)
# dyna_params = torch.tensor([0.191, 0.212, 0.01, 0.01, 0.798, 0.122], device=device)
dyna_params = torch.ones(6, device=device)
dynamic_system = SystemModel(tableDampingX=dyna_params[0], tableDampingY=dyna_params[1],
                             tableFrictionX=dyna_params[2],
                             tableFrictionY=dyna_params[3],
                             tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                             puckRadius=0.03165, malletRadius=0.04815,
                             tableRes=dyna_params[4],
                             malletRes=0.8, rimFriction=dyna_params[5],
                             dt=1 / 120, device=device)
puck_EKF = AirHockeyEKF(u=1 / 120., system=dynamic_system, Q=Q, R=R, P=P, device=device,
                        Q_collision=Q_collision)
if save:
    writer = SummaryWriter(save_path)
    save_gap = 1
for time in tqdm(range(epoch)):
    total_filter_state_list = []
    total_predict_state_list = []
    total_coll_mode_list = []
    total_transform_jacobian_list = []
    total_slide_list = []
    total_trajectory = []
    non_coll_list = []
    for i in range(len(total_dataset)):
        trajectory = total_dataset[i]
        trajectory = torch.tensor(trajectory, device=device).float()
        init_state = calculate_init_state(trajectory, device=device)
        if filter_type == 'EKF':
            filter_state, collision_list, _, _, _, predict_state, _, _, trans_jacobian_list, coll_mode_list, slide_list, _, _ = puck_EKF.kalman_filter(
                init_state, trajectory[1:], coll_mode=True)
        elif filter_type == 'smooth':
            filter_state, collision_list, _, predict_state, trans_jacobian_list, coll_mode_list, slide_list = puck_EKF.smooth(
                init_state, trajectory[1:], coll_mode=True)
        if np.any(collision_list):
            continue
        non_coll_list.append(i)
        total_predict_state_list += predict_state
        total_filter_state_list += filter_state
        total_coll_mode_list += coll_mode_list
        total_transform_jacobian_list += trans_jacobian_list
        total_slide_list += slide_list
        total_trajectory.append(trajectory[1:])
    total_trajectory = torch.vstack(total_trajectory)
    # Linear Regression calculate dynamics parameters
    # needed: input filter state, output filter state and dynamics type:non_collision or which mode collision
    new_dynamic_params = linear_regression_dynamics_params(total_filter_state_list, total_coll_mode_list,
                                                           total_dataset[non_coll_list],
                                                           total_transform_jacobian_list, total_slide_list,
                                                           device=device)
    puck_EKF.system.set_params(tableDampingX=new_dynamic_params[0], tableDampingY=new_dynamic_params[1],
                               restitution=new_dynamic_params[2], rimFriction=new_dynamic_params[3])
    # calculate new R Q and Q_collision
    # needed: filter state / predict state and recorded state with correspond dynamics type
    R, Q, Q_collision = calculate_variance(total_filter_state_list, total_predict_state_list, total_trajectory,
                                           total_coll_mode_list, total_transform_jacobian_list)
    puck_EKF.refresh(Q=Q, Q_collision=Q_collision, R=R)
    if save and time % save_gap == 0:
        writer.add_scalar('dynamics/table damping x', new_dynamic_params[0], time)
        writer.add_scalar('dynamics/table damping y', new_dynamic_params[1], time)
        writer.add_scalar('dynamics/table restitution', new_dynamic_params[2], time)
        writer.add_scalar('dynamics/table rimfriction', new_dynamic_params[3], time)
        writer.add_scalar('observation_variance/x', R[0, 0], time)
        writer.add_scalar('observation_variance/y', R[1, 1], time)
        writer.add_scalar('observation_variance/theta', R[2, 2], time)
        writer.add_scalar('dynamics_non_collision_variance/x', Q[0, 0], time)
        writer.add_scalar('dynamics_non_collision_variance/y', Q[1, 1], time)
        writer.add_scalar('dynamics_non_collision_variance/vx', Q[2, 2], time)
        writer.add_scalar('dynamics_non_collision_variance/vy', Q[3, 3], time)
        writer.add_scalar('dynamics_non_collision_variance/theta', Q[4, 4], time)
        writer.add_scalar('dynamics_non_collision_variance/dtheta', Q[5, 5], time)
        writer.add_scalar('dynamics_collision_variance/x', Q_collision[0, 0], time)
        writer.add_scalar('dynamics_collision_variance/y', Q_collision[1, 1], time)
        writer.add_scalar('dynamics_collision_variance/vx', Q_collision[2, 2], time)
        writer.add_scalar('dynamics_collision_variance/vy', Q_collision[3, 3], time)
        writer.add_scalar('dynamics_collision_variance/theta', Q_collision[4, 4], time)
        writer.add_scalar('dynamics_collision_variance/dtheta', Q_collision[5, 5], time)

import numpy as np
import torch
from matplotlib import pyplot as plt
from math import pi

device = torch.device("cuda")


def calculate_init_state(trajectory):
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


# input: state_list:calculated by EKF and kalman smooth, correspond trajectory
# output: draw the trajectory and position, velocity
# color: r smooth b EKF g data
def plot_with_state_list(EKF_state_list, smooth_state_list, trajectory, time_list, writer=None, epoch=0,
                         trajectory_index=None):
    # EKF_state_list = torch.tensor([item.clone().cpu().numpy() for item in EKF_state_list], device=device).cpu().numpy()
    EKF_state_list = torch.stack(EKF_state_list).cpu().numpy()
    # smooth_state_list = torch.tensor([item.clone().cpu().numpy() for item in smooth_state_list], device=device).cpu().numpy()
    smooth_state_list = torch.stack(smooth_state_list).cpu().numpy()
    trajectory = trajectory.cpu().numpy()
    x_velocity = []
    y_velocity = []
    theta_velocity = []
    for i in range(1, len(trajectory)):
        x_velocity.append((trajectory[i][0] - trajectory[i - 1][0]) / (trajectory[i][3] - trajectory[i - 1][3]))
        y_velocity.append((trajectory[i][1] - trajectory[i - 1][1]) / (trajectory[i][3] - trajectory[i - 1][3]))
        if abs(trajectory[i][2] - trajectory[i - 1][2]) > pi:
            theta_velocity.append(
                (trajectory[i][2] - np.sign(trajectory[i][2]) * pi) / (trajectory[i][-1] - trajectory[i - 1][-1]))
        else:
            theta_velocity.append(
                (trajectory[i][2] - trajectory[i - 1][2]) / (trajectory[i][-1] - trajectory[i - 1][-1]))
    plt.figure()
    plt.scatter(trajectory[1:, 0], trajectory[1:, 1], c='g', label='recorded trajectory', alpha=0.5, s=2)
    plt.scatter(EKF_state_list[:, 0], EKF_state_list[:, 1], c='b', label='EKF trajectory', alpha=0.5, s=2)
    plt.scatter(smooth_state_list[:, 0], smooth_state_list[:, 1], c='r', label='Smooth trajectory', s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + "/cartesian", plt.gcf(), epoch)
    # position x
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x position')
    plt.scatter(time_list, EKF_state_list[:, 0], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 0], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 0], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    # position y
    plt.subplot(3, 1, 2)
    plt.title('y position')
    plt.scatter(time_list, EKF_state_list[:, 1], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 1], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 1], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('theta')
    plt.scatter(time_list, EKF_state_list[:, 4], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 2], label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 4], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/position', plt.gcf(), epoch)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x velocity')
    plt.scatter(time_list, EKF_state_list[:, 2], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], x_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 2], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('y velocity')
    plt.scatter(time_list, EKF_state_list[:, 3], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], y_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 3], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('rotate velocity')
    plt.scatter(time_list, EKF_state_list[:, 5], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], theta_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], smooth_state_list[-1::-1, 5], label='Smooth trajectory', c='r',
                s=2)
    plt.legend()
    if writer != None:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/velocity', plt.gcf(), epoch)
    plt.close()
    # plt.show()


def EKF_plot_with_state_list(EKF_state_list, trajectory, prediction=None, writer=None, epoch=0, trajectory_index=None, tag='', save_gap=5):
    # EKF_state_list = torch.tensor([item.clone().cpu().numpy() for item in EKF_state_list], device=device).cpu().numpy()
    EKF_state_list = torch.stack(EKF_state_list).cpu().numpy()
    prediction = torch.stack(prediction).cpu().numpy()
    trajectory = trajectory.cpu().numpy()
    x_velocity = []
    y_velocity = []
    theta_velocity = []
    for i in range(1, len(trajectory)):
        x_velocity.append((trajectory[i][0] - trajectory[i - 1][0]) / (trajectory[i][3] - trajectory[i - 1][3]))
        y_velocity.append((trajectory[i][1] - trajectory[i - 1][1]) / (trajectory[i][3] - trajectory[i - 1][3]))
        if abs(trajectory[i][2] - trajectory[i - 1][2]) > pi:
            theta_velocity.append(
                (trajectory[i][2] - np.sign(trajectory[i][2]) * pi) / (trajectory[i][-1] - trajectory[i - 1][-1]))
        else:
            theta_velocity.append(
                (trajectory[i][2] - trajectory[i - 1][2]) / (trajectory[i][-1] - trajectory[i - 1][-1]))
    fig = plt.figure()
    xy = [0, -1.038 / 2]
    ax = fig.add_subplot(111)
    rect = plt.Rectangle(xy, 1.948, 1.038, fill=False)
    rect.set_linewidth(10)
    ax.add_patch(rect)
    time_list = []
    for i in range(len(EKF_state_list)):
        time_list.append(i/120);
    plt.scatter(trajectory[1:, 0], trajectory[1:, 1], c='g', label='recorded trajectory', alpha=0.5, s=2)
    plt.scatter(EKF_state_list[:, 0], EKF_state_list[:, 1], c='b', label='EKF trajectory', alpha=0.5, s=2)
    plt.scatter(prediction[:, 0], prediction[:, 1], c='r', label='predicted trajectory', alpha=0.5, s=2)
    plt.legend()
    if writer != None and epoch % save_gap == 0:
        writer.add_figure('trajectory_' + str(trajectory_index) + "/cartesian"+tag, plt.gcf(), epoch)
    plt.close()
    # position x
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x position')
    plt.scatter(time_list, EKF_state_list[:, 0], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 0], label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 0], label='predict trajectory', c='r', s=2)
    plt.legend()
    # position y
    plt.subplot(3, 1, 2)
    plt.title('y position')
    plt.scatter(time_list, EKF_state_list[:, 1], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 1], label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 1], label='predict trajectory', c='r', s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('theta')
    plt.scatter(time_list, EKF_state_list[:, 4], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:, 3] - trajectory[0, 3], trajectory[:, 2], label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 4], label='predict trajectory', c='r', s=2)
    plt.legend()
    if writer != None and epoch % save_gap == 0:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/position'+tag, plt.gcf(), epoch)
    plt.close()
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title('x velocity')
    plt.scatter(time_list, EKF_state_list[:, 2], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], x_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 2], label='predict trajectory', c='r', s=2)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title('y velocity')
    plt.scatter(time_list, EKF_state_list[:, 3], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], y_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 3], label='predict trajectory', c='r', s=2)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.title('rotate velocity')
    plt.scatter(time_list, EKF_state_list[:, 5], label='EKF trajectory', c='b', s=2)
    plt.scatter(trajectory[:-1, 3] - trajectory[0, 3], theta_velocity, label='recorded trajectory', c='g', s=2)
    plt.scatter(time_list, prediction[:, 5], label='predict trajectory', c='r', s=2)
    plt.legend()
    if writer != None and epoch % save_gap == 0:
        writer.add_figure('trajectory_' + str(trajectory_index) + '/velocity'+tag, plt.gcf(), epoch)
    plt.close()



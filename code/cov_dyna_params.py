import torch
from torch_air_hockey_baseline_no_detach import SystemModel


def generate_params(device):
    covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 5e-4, 5, 0.0225, 225]).to(device=device)
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 4e-10, 4e-6, 1e-6, 0.01]).to(device=device)
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
    P = torch.eye(6, device=device) * 0.01
    dyna_params = torch.tensor([0.191, 0.212, 0.01, 0.01, 0.798, 0.122], device=device)
    dynamic_system = SystemModel(tableDampingX=dyna_params[0], tableDampingY=dyna_params[1],
                                 tableFrictionX=dyna_params[2],
                                 tableFrictionY=dyna_params[3],
                                 tableLength=1.948, tableWidth=1.038, goalWidth=0.25,
                                 puckRadius=0.03165, malletRadius=0.04815,
                                 tableRes=dyna_params[4],
                                 malletRes=0.8, rimFriction=dyna_params[5],
                                 dt=1 / 120, device=device)
    return P, R, Q, Q_collision, dyna_params, dynamic_system

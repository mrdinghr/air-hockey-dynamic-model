import argparse
import os
from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment
from experiment_launcher.launcher import add_launcher_base_args, get_experiment_default_params
from statedependentparams import FullResState
import datetime
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch_gradient import Kalman_EKF_Gradient
from torch_gradient import load_dataset


# This decorator is not mandatory.
# It creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment
def experiment(
        save_gap: int = 5,
        lr: float = 1e-5,
        plot: bool = False,
        epoch: int = 200,
        seed: int = 0,
        results_dir: str = '/tmp',
        loss_type: str = ''
):
    file_name = 'hundred_data_after_clean.npy'
    training_dataset, test_dataset = load_dataset(file_name)
    device = torch.device('cpu')
    batch_size = 32
    batch_trajectory_size = 10
    full_res = FullResState(device=device)
    # full_res.load_state_dict(torch.load('./alldata/EKF+EKF+full_log_like+only_collision_nn+1105/model.pt'))
    full_res.to(device)
    init_params = torch.tensor([0.2, 0.2, 0.01, 0.01, 0.798, 0.122], device=device)
    covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 5e-4, 5, 0.0225, 6.25]).to(device=device)
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 4e-10, 4e-6, 1e-6, 0.01]).to(device=device)
    model = Kalman_EKF_Gradient(params=init_params, covariance_params=covariance_params,
                                segment_size=batch_trajectory_size, device=device,
                                covariance_params_collision=covariance_params_collision)
    optimizer = torch.optim.Adam(full_res.parameters(), lr=lr)
    prepare_typ = 'EKF'
    loss_form = 'EKF'
    addition_information = '+only_collision_nn+2mode_collision+1106'
    logdir = './alldata/' + prepare_typ + '+' + loss_form + '+' + loss_type + addition_information
    writer = SummaryWriter(logdir)
    pre_loss = 2000
    for t in tqdm(range(epoch)):
        # params: damping x, damping y, friction x, friction y, restitution, rimfriction
        beta = 15
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=plot,
                                                         type=prepare_typ, beta=beta, full_res=full_res, save_gap=save_gap)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type=loss_form,
                                        epoch=t, beta=beta, full_res=full_res, loss_type=loss_type)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.detach().cpu().numpy())
            else:
                batch_loss.append(loss.cpu().numpy())
        training_loss = np.mean(batch_loss)
        writer.add_scalar('loss/training_loss', training_loss, t)
        test_segment_dataset = model.prepare_dataset(test_dataset, type=prepare_typ, epoch=t, beta=beta,
                                                     full_res=full_res, plot=plot, writer=writer, tag='test', save_gap=save_gap)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type=loss_form, epoch=t,
                                            beta=beta, full_res=full_res, loss_type=loss_type)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)
        if t % save_gap == 0:
            torch.save(full_res.state_dict(), logdir + "/model.pt")
        if ('mse' in loss_type and (training_loss >= pre_loss + 10 or training_loss >= 1.2 * pre_loss)) or ('log_like' in loss_type and training_loss > pre_loss + 20):
            torch.save(full_res.state_dict(), logdir + "/strange_"+str(t)+"_model.pt")
        pre_loss = training_loss


# You can specify your own parser, or use the experiment_launcher parser.
def parse_args():
    parser = argparse.ArgumentParser()

    # Place your experiment arguments here
    arg_test = parser.add_argument_group('Test')
    arg_test.add_argument("--env", type=str)
    arg_test.add_argument("--env-param", type=str)
    arg_test.add_argument("--a", type=int)
    arg_test.add_argument("--boolean-param", action='store_true')
    arg_test.add_argument('--some-default-param', type=str)

    # Leave unchanged
    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)

    # To use your own parser, run instead.
    # args = parse_args()
    # run_experiment(experiment, args)

import datetime
import pdb
import numpy as np
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch_gradient import Kalman_EKF_Gradient
from torch_gradient import load_dataset

torch.set_printoptions(threshold=torch.inf)


class StateDependentParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 6)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)

    def cal_params(self, state):
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        # output = torch.nn.functional.relu(output)
        output = torch.sigmoid(output)
        return output


class ResState(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = torch.nn.Linear(6, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.fc5 = torch.nn.Linear(256, 3)
        self.fc6 = torch.nn.Linear(256, 256)
        self.fc7 = torch.nn.Linear(256, 3)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc5.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc6.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc7.weight, gain=1.0)
        # torch.nn.init.constant_(self.fc1.weight, 0)
        # torch.nn.init.constant_(self.fc2.weight, 0)
        # torch.nn.init.constant_(self.fc3.weight, 0)
        # torch.nn.init.constant_(self.fc4.weight, 0)
        # torch.nn.init.constant_(self.fc5.weight, 0)
        # torch.nn.init.constant_(self.fc6.weight, 0)
        # torch.nn.init.constant_(self.fc7.weight, 0)
        # torch.nn.init.constant_(self.fc1.bias, 0)
        # torch.nn.init.constant_(self.fc2.bias, 0)
        # torch.nn.init.constant_(self.fc3.bias, 0)
        # torch.nn.init.constant_(self.fc4.bias, 0)
        # torch.nn.init.constant_(self.fc5.bias, 0)
        # torch.nn.init.constant_(self.fc6.bias, 0)
        # torch.nn.init.constant_(self.fc7.bias, 0)

    def cal_res(self, input):
        state = input
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc5(output)
        output = 120 * torch.tensor([4e-3, 4e-3, 0.2], device=self.device) * torch.tanh(output)
        # output = torch.nn.functional.relu(output)
        return output

    def cal_res_collision(self, input):
        state = input
        # state = input[[0, 1, 2, 3, 5]]
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc6(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc7(output)
        output = 120 * torch.tensor([1., 1., 30], device=self.device) * torch.tanh(output)
        # output = torch.tensor([0.0, 0.0, 1., 1., 0.0, 5], device=device) * torch.tanh(output)
        return output


class FullResState(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = torch.nn.Linear(6, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.fc5 = torch.nn.Linear(256, 6)
        self.fc6 = torch.nn.Linear(256, 256)
        self.fc7 = torch.nn.Linear(256, 6)
        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc4.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc5.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc6.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.fc7.weight, gain=1.0)

    def cal_res(self, input):
        state = input
        # state = input[[0, 1, 2, 3, 5]]
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc4(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc5(output)
        output = torch.tensor([4e-5, 4e-5, 4e-3, 4e-3, 2e-3, 0.2], device=self.device) * torch.tanh(output)
        return output

    def cal_res_collision(self, input):
        state = input
        # state = input[[0, 1, 2, 3, 5]]
        output = self.fc1(state)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc6(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.fc7(output)
        # 0.005 0.005 1 1 0.3 30
        output = torch.tensor([0.005, 0.005, 1, 1, 0.3, 5], device=self.device) * torch.tanh(output)
        return output


class FixedParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("dyna_params", torch.nn.Parameter(torch.tensor([0.2, 0.2, 0.1, 0.1, 0.798, 0.122])))

    def cal_params(self, state):
        if len(state.size()) == 1:
            return torch.abs(self.dyna_params)
        elif len(state.size()) == 2:
            return torch.abs(self.dyna_params.tile(state.shape[-2], 1))


if __name__ == '__main__':
    file_name = 'hundred_data_one_coll.npy'
    training_dataset, test_dataset = load_dataset(file_name)
    torch.manual_seed(0)
    device = torch.device("cuda")  # cuda cpu
    lr = 1e-5
    batch_size = 32
    batch_trajectory_size = 10
    epochs = 2000
    save_gap = 1
    plot = False
    cal = None
    # cal = StateDependentParams()
    # cal = FixedParams()
    # cal.to(device)
    res = None
    # res = ResState(device=device)
    # res.load_state_dict(torch.load('./alldata/819nn/2022-08-19-14-07-08EKF+EKF+xyomega_log_like+twonet+onemodecoll+nnxddt+moretrajectories/model.pt'))
    # res.to(device)
    full_res = FullResState(device=device)
    full_res.load_state_dict(torch.load('./alldata/EKF+EKF+full_log_like+1101/model.pt'))
    full_res.to(device)
    # params: damping x, damping y, friction x, friction y, restitution, rimfriction
    if cal is not None:
        init_params = cal.cal_params(torch.tensor([0., 0., 0., 0., 0., 0.], device=device))
    else:
        init_params = torch.tensor([0.2, 0.2, 0.01, 0.01, 0.798, 0.122], device=device)
    #  R0， R1， R2， Q01， Q23，Q4， Q5  R observation noise  Q dynamic noise
    # covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 2e-10, 1e-7, 1.0e-2, 1.0e-1]).to(device=device) # original variance
    # covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 25e-6, 25e-2, 6.25e-4, 6.25]).to(device=device) # original collision variance
    covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 25e-6, 25e-2, 0.0225, 225]).to(
        device=device)  # res collision variance
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 1e-8, 1e-4, 1e-6, 0.01]).to(
        device=device)  # res variance
    # bigger dynamic variance
    covariance_params_collision = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 5e-4, 5, 0.0225, 225]).to(device=device)
    covariance_params = torch.Tensor([2.5e-7, 2.5e-7, 9.1e-3, 4e-10, 4e-6, 1e-6, 0.01]).to(device=device)
    model = Kalman_EKF_Gradient(params=init_params, covariance_params=covariance_params,
                                segment_size=batch_trajectory_size, device=device,
                                cal=cal, covariance_params_collision=covariance_params_collision)
    if res is not None:
        optimizer = torch.optim.Adam(res.parameters(), lr=lr)
    elif cal is not None:
        optimizer = torch.optim.Adam(cal.parameters(), lr=lr)
    elif full_res is not None:
        optimizer = torch.optim.Adam(full_res.parameters(), lr=lr)
    prepare_typ = 'EKF'
    loss_form = 'EKF'  # EKF predict
    loss_type = 'full_log_like'  # mse log_like  multi_mse  multi_log_like  xyomega_log_like  xyomega_mse vomega_log_like vomega_mse full_log_like full_mse
    addition_information = '+debug'  # moretrajectories  onetrajectory
    logdir = './alldata/919nn' + datetime.datetime.now().strftime(
        "/%Y-%m-%d-%H-%M-%S") + prepare_typ + '+' + loss_form + '+' + loss_type + addition_information
    writer = SummaryWriter(logdir)
    pre_loss = 2000
    for t in tqdm(range(epochs)):
        # params: damping x, damping y, friction x, friction y, restitution, rimfriction
        # beta = 29 * t / epochs + 1
        beta = 15
        training_segment_dataset = model.prepare_dataset(training_dataset, epoch=t, writer=writer, plot=plot,
                                                         type=prepare_typ, cal=cal,
                                                         beta=beta, res=res, save_gap=save_gap, full_res=full_res)
        training_index_list = range(len(training_segment_dataset))
        loader = Data.DataLoader(training_index_list, batch_size=batch_size, shuffle=True)
        batch_loss = []
        if cal is not None:
            params = torch.abs(cal.dyna_params)
            # params = cal.cal_params(training_segment_dataset[:, 2:4]).mean(dim=0)
        else:
            params = model.params
        if t % save_gap == 0 and cal is not None:
            writer.add_scalar('dynamics/table damping x', params[0], t)
            writer.add_scalar('dynamics/table damping y', params[1], t)
            writer.add_scalar('dynamics/table friction x', params[2], t)
            writer.add_scalar('dynamics/table friction y', params[3], t)
            writer.add_scalar('dynamics/table restitution', params[4], t)
            writer.add_scalar('dynamics/rim friction', params[5], t)
        for index_batch in tqdm(loader):
            optimizer.zero_grad()
            loss = model.calculate_loss(training_segment_dataset[index_batch], training_dataset, type=loss_form,
                                        epoch=t, cal=cal, beta=beta, res=res, loss_type=loss_type, full_res=full_res)
            if loss.requires_grad:
                loss.backward()
                print("loss:", loss.item())
                # print("dynamics: ", model.params.detach().cpu().numpy())
                optimizer.step()
                batch_loss.append(loss.detach().cpu().numpy())
            else:
                print("loss:", loss.item())
                # print("dynamics: ", model.params.detach().cpu().numpy())
                batch_loss.append(loss.cpu().numpy())
        training_loss = np.mean(batch_loss)
        writer.add_scalar('loss/training_loss', training_loss, t)
        test_segment_dataset = model.prepare_dataset(test_dataset, type=prepare_typ, epoch=t, cal=cal, beta=beta,
                                                     res=res, plot=plot, writer=writer, tag='test', save_gap=save_gap,
                                                     full_res=full_res)
        test_index_list = range(len(test_segment_dataset))
        test_loader = Data.DataLoader(test_index_list, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            test_batch_loss = []
            for index_batch in tqdm(test_loader):
                loss = model.calculate_loss(test_segment_dataset[index_batch], test_dataset, type=loss_form, epoch=t,
                                            cal=cal, beta=beta, res=res, loss_type=loss_type, full_res=full_res)
                test_batch_loss.append(loss.detach().cpu().numpy())
            test_loss = np.mean(test_batch_loss)
            writer.add_scalar('loss/test_loss', test_loss, t)

        if t % save_gap == 0 or ('mse' in loss_type and training_loss >= 5 * pre_loss) or ('log_like' in loss_type and training_loss > pre_loss + 50):
            if cal is not None:
                torch.save(cal.state_dict(), logdir + "/model.pt")
            if res is not None:
                torch.save(res.state_dict(), logdir + "/model.pt")
            if full_res is not None:
                torch.save(full_res.state_dict(), logdir + "/model.pt")
        pre_loss = training_loss

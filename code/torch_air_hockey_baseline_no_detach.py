from math import pi

import numpy as np
import torch

b_params = np.load('./dyna_params/coll_params_b.npy')
n_params = np.load('./dyna_params/coll_params_n.npy')
theta_params = np.load('./dyna_params/coll_params_theta.npy')
b_params = torch.from_numpy(b_params).type(torch.FloatTensor)
n_params = torch.from_numpy(n_params).type(torch.FloatTensor)
theta_params = torch.from_numpy(theta_params).type(torch.FloatTensor)
non_x_params = np.load('./dyna_params/non_coll_params_b.npy')
non_y_params = np.load('./dyna_params/non_coll_params_n.npy')
non_theta_params = np.load('./dyna_params/non_coll_params_theta.npy')
non_x_params = torch.from_numpy(non_x_params).type(torch.FloatTensor)
non_y_params = torch.from_numpy(non_y_params).type(torch.FloatTensor)
non_theta_params = torch.from_numpy(non_theta_params).type(torch.FloatTensor)


def cross2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


class AirHockeyTable:
    def __init__(self, length, width, goalWidth, puckRadius, restitution, rimFriction, dt, tableDampingX,
                 tableDampingY, device):
        self.m_length = length
        self.m_width = width
        self.m_puckRadius = puckRadius
        self.m_goalWidth = goalWidth
        self.m_e = restitution
        self.m_rimFriction = rimFriction
        self.m_dt = dt
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY
        self.device = device

        ref = torch.tensor([length / 2, 0.])
        offsetP1 = torch.tensor([-self.m_length / 2 + self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP2 = torch.tensor([-self.m_length / 2 + self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP3 = torch.tensor([self.m_length / 2 - self.m_puckRadius, -self.m_width / 2 + self.m_puckRadius])
        offsetP4 = torch.tensor([self.m_length / 2 - self.m_puckRadius, self.m_width / 2 - self.m_puckRadius])
        offsetP1 = offsetP1 + ref
        offsetP2 = offsetP2 + ref
        offsetP3 = offsetP3 + ref
        offsetP4 = offsetP4 + ref
        self.m_boundary = torch.tensor([[offsetP1[0], offsetP1[1], offsetP3[0], offsetP3[1]],
                                        [offsetP3[0], offsetP3[1], offsetP4[0], offsetP4[1]],
                                        [offsetP4[0], offsetP4[1], offsetP2[0], offsetP2[1]],
                                        [offsetP2[0], offsetP2[1], offsetP1[0], offsetP1[1]]], device=device)

        self.m_jacCollision = torch.eye(6, device=device)
        # transform matrix from global to local
        #   First Rim
        T_tmp = torch.eye(6, device=device)
        self.m_rimGlobalTransforms = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransformsInv = torch.zeros((4, 6, 6), device=device)
        self.m_rimGlobalTransforms[0] = T_tmp
        self.m_rimGlobalTransformsInv[0] = torch.linalg.inv(T_tmp)
        #   Second Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = 1
        T_tmp[1][0] = -1
        T_tmp[2][3] = 1
        T_tmp[3][2] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[1] = T_tmp
        self.m_rimGlobalTransformsInv[1] = torch.linalg.inv(T_tmp)
        #   Third Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][0] = -1
        T_tmp[1][1] = -1
        T_tmp[2][2] = -1
        T_tmp[3][3] = -1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[2] = T_tmp
        self.m_rimGlobalTransformsInv[2] = torch.linalg.inv(T_tmp)
        #   Forth Rim
        T_tmp = torch.zeros((6, 6), device=device)
        T_tmp[0][1] = -1
        T_tmp[1][0] = 1
        T_tmp[2][3] = -1
        T_tmp[3][2] = 1
        T_tmp[4][4] = 1
        T_tmp[5][5] = 1
        self.m_rimGlobalTransforms[3] = T_tmp
        self.m_rimGlobalTransformsInv[3] = T_tmp.T

    def set_dynamic_parameter(self, restitution, rimFriction, tableDampingX, tableDampingY):
        self.m_e = restitution
        self.m_rimFriction = rimFriction
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY

    def close_boundary(self, state):
        if abs(state[0] - self.m_boundary[0, 0]) < 0.05 or abs(state[0] - self.m_boundary[1, 0]) < 0.05 or abs(
                state[1] - self.m_boundary[0, 1]) < 0.05 or abs(state[1] - self.m_boundary[1, 3]) < 0.05:
            return True

    def collision_outof_boundary(self, i, pos, vel):
        if i == 0:
            if pos[1] <= self.m_boundary[i][1] and vel[1] < 0:
                return True
            else:
                return False
        if i == 1:
            if pos[0] >= self.m_boundary[i][0] and vel[0] > 0:
                return True
            else:
                return False
        if i == 2:
            if pos[1] >= self.m_boundary[i][1] and vel[1] > 0:
                return True
            else:
                return False
        if i == 3:
            if pos[0] <= self.m_boundary[i][0] and vel[0] < 0:
                return True
            else:
                return False

    def collision_in_boundary(self, s, r, pos):
        if ((self.m_boundary[2][:2] - pos > 0).all() and (self.m_boundary[0][:2] - pos < 0).all() and (
                s >= 1e-4 and s <= 1 - 1e-4 and r >= 1e-4 and r <= 1 - 1e-4)):
            return True
        else:
            return False

    def apply_collision(self, state, Q_collision=None, beta=1, save_weight=False, writer=None, epoch=0,
                        collision_time=0, coll_mode=False, cal_mode='spong'):
        pos = state[0:2]
        vel = state[2:4]
        angle = state[4]
        ang_vel = state[5]
        score = False
        if torch.abs(pos[1]) < self.m_goalWidth / 2 and pos[0] < self.m_boundary[0][0] + self.m_puckRadius:
            score = True
        elif torch.abs(pos[1]) < self.m_goalWidth / 2 and pos[0] > self.m_boundary[0][2] - self.m_puckRadius:
            score = True
        u = vel * self.m_dt
        cur_state = torch.zeros(6, device=self.device)
        for i in range(self.m_boundary.shape[0]):
            p1 = self.m_boundary[i][0:2]
            p2 = self.m_boundary[i][2:]
            v = p2 - p1  # torch.tensor([p2[0] - p1[0], p2[1] - p1[1]], device=device).double()
            w = p1 - pos  # torch.tensor([p1[0] - p[0], p1[1] - p[1]], device=device).double()
            # denominator = cross2d(v, u.detach())
            denominator = cross2d(v, u)
            if abs(denominator) < 1e-6:
                continue
            s = cross2d(v, w) / denominator
            # r = cross2d(u.detach(), w) / denominator
            r = cross2d(u, w) / denominator
            if self.collision_in_boundary(s, r, pos) or self.collision_outof_boundary(i, pos, vel):
                if self.collision_outof_boundary(i, pos, vel):
                    s = 0
                if Q_collision is not None:
                    Q_collision = self.m_rimGlobalTransformsInv[i] @ Q_collision @ self.m_rimGlobalTransformsInv[i].T
                state_pre = pos + s * u
                theta_pre = angle + s * ang_vel * self.m_dt
                F_pre_collision = torch.eye(6, device=self.device)
                F_pre_collision[0][2] = s * self.m_dt
                F_pre_collision[1][3] = s * self.m_dt
                F_pre_collision[4][5] = s * self.m_dt
                F_post_collision = torch.eye(6, device=self.device)
                F_post_collision[0][2] = (1 - s) * self.m_dt
                F_post_collision[1][3] = (1 - s) * self.m_dt
                F_post_collision[4][5] = (1 - s) * self.m_dt
                vecT = v / torch.linalg.norm(v)
                vecN = torch.stack([-v[1] / torch.linalg.norm(v), v[0] / torch.linalg.norm(v)]).to(device=self.device)
                vtScalar = torch.dot(vel, vecT)
                vnSCalar = torch.dot(vel, vecN)
                weight = 0
                if save_weight:
                    writer.add_scalar('weight/weight' + str(collision_time), weight, epoch)
                    collision_time += 1
                slideDir = torch.sign(vtScalar + ang_vel * self.m_puckRadius)
                if cal_mode == 'spong':
                    if 3 * self.m_rimFriction * (1 + self.m_e) * torch.abs(vnSCalar) - torch.abs(
                            vtScalar + self.m_puckRadius * ang_vel) > 0:
                        m_jacCollision_mode_no_slide = torch.eye(6, device=self.device)
                        m_jacCollision_mode_no_slide[2][2] = 2 / 3
                        m_jacCollision_mode_no_slide[2][5] = -self.m_puckRadius / 3
                        m_jacCollision_mode_no_slide[3][3] = -self.m_e
                        m_jacCollision_mode_no_slide[5][2] = -2 / (3 * self.m_puckRadius)
                        m_jacCollision_mode_no_slide[5][5] = 1 / 3
                        m_jacCollision = m_jacCollision_mode_no_slide
                        mode = 'no_slide'
                    else:
                        mu = self.m_rimFriction
                        # mu = min(self.m_rimFriction,
                        #           abs(vtScalar + self.m_puckRadius * ang_vel) / (3 * (1 + self.m_e) * torch.abs(vnSCalar)))
                        m_jacCollision_mode_slide = torch.eye(6, device=self.device)
                        m_jacCollision_mode_slide[2][3] = mu * slideDir * (1 + self.m_e)
                        m_jacCollision_mode_slide[3][3] = -self.m_e
                        m_jacCollision_mode_slide[5][3] = mu * slideDir * (1 + self.m_e) * 2 / self.m_puckRadius
                        m_jacCollision = m_jacCollision_mode_slide
                        mode = 'slide'
                    # elif cal_mode == 'statistik':
                    #     m_jacCollision = torch.eye(6, device=self.device)
                    #     m_jacCollision[2, [2, 3, 5]] = b_params[0:3]
                    #     m_jacCollision[3, [2, 3, 5]] = n_params[0:3]
                    #     m_jacCollision[5, [2, 3, 5]] = theta_params[0:3]
                    # m_jacCollision = weight * m_jacCollision_mode_no_slide + (1 - weight) * m_jacCollision_mode_slide
                    jacobian_global = self.m_rimGlobalTransformsInv[i] @ m_jacCollision @ self.m_rimGlobalTransforms[i]
                    jacobian = F_post_collision @ jacobian_global @ F_pre_collision
                    # if jacobian[4] @ state > pi:
                    #     cur_state4 = jacobian[4] @ state - 2 * pi
                    # elif jacobian[4] @ state < -pi:
                    #     cur_state4 = jacobian[4] @ state + 2 * pi
                    # else:
                    #     # cur_state[4] = theta_pre + (1 - s) * cur_state5 * self.m_dt
                    #     cur_state4 = jacobian[4] @ state
                    cur_state4 = jacobian[4] @ state
                    cur_state = torch.cat(
                        (jacobian[0:4] @ state, torch.atleast_1d(cur_state4), torch.atleast_1d(jacobian[5] @ state)))
                elif cal_mode == 'statistik':
                    m_jacCollision = torch.eye(6, device=self.device)
                    m_jacCollision[2, [2, 3, 5]] = b_params[0:3]
                    m_jacCollision[2, 3] = m_jacCollision[2, 3] * slideDir
                    m_jacCollision[3, [2, 3, 5]] = n_params[0:3]
                    m_jacCollision[5, [2, 3, 5]] = theta_params[0:3]
                    m_jacCollision[5, 3] = m_jacCollision[5, 3] * slideDir
                    jacobian_global = self.m_rimGlobalTransformsInv[i] @ m_jacCollision @ self.m_rimGlobalTransforms[i]
                    jacobian = F_post_collision @ jacobian_global @ F_pre_collision
                    cur_state = jacobian @ state
                    mode = cal_mode
                if coll_mode:
                    return True, cur_state, jacobian, Q_collision, self.m_rimGlobalTransforms[i], mode, slideDir
                return True, cur_state, jacobian, Q_collision
        if coll_mode:
            return False, state, torch.eye(6, device=self.device), Q_collision, torch.eye(6, device=self.device), '', 0
        return False, state, torch.eye(6, device=self.device), Q_collision


class SystemModel:
    def __init__(self, tableDampingX, tableDampingY, tableFrictionX, tableFrictionY, tableLength, tableWidth, goalWidth,
                 puckRadius, malletRadius,
                 tableRes, malletRes, rimFriction, dt, device):
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY
        self.tableFrictionX = tableFrictionX
        self.tableFrictionY = tableFrictionY
        self.tableLength = tableLength
        self.tableWidth = tableWidth
        self.goalWidth = goalWidth
        self.puckRadius = puckRadius
        self.malletRadius = malletRadius
        self.dt = dt
        self.table = AirHockeyTable(length=tableLength, width=tableWidth, goalWidth=goalWidth,
                                    puckRadius=puckRadius, restitution=tableRes, rimFriction=rimFriction, dt=dt,
                                    tableDampingX=tableDampingX, tableDampingY=tableDampingY, device=device)
        self.device = device

    @property
    def F(self):
        J_linear = torch.eye(6, device=self.device)
        J_linear[0][2] = self.dt
        J_linear[1][3] = self.dt
        J_linear[2][2] = 1 - self.dt * self.tableDampingX
        J_linear[3][3] = 1 - self.dt * self.tableDampingY
        J_linear[4][5] = self.dt
        J_linear[5][5] = 1
        return J_linear

    def f(self, x, u, cal_mode='spong'):
        pos_prev = x[0:2]
        vel_prev = x[2:4]
        ang_prev = x[4]
        ang_vel_prev = x[5]
        pos = pos_prev + u * vel_prev
        angle = ang_prev + u * ang_vel_prev
        if cal_mode == 'spong':
            vel = vel_prev - u * torch.stack([self.tableDampingX, self.tableDampingY]) * vel_prev
            ang_vel = ang_vel_prev
        elif cal_mode == 'statistik':
            vel = vel_prev - u * torch.stack([self.tableDampingX, self.tableDampingY]) * vel_prev
            ang_vel = ang_vel_prev
            # vel = torch.zeros(2)
            # vel[0] = non_x_params[0] * x[2] + non_x_params[1] * x[3] + non_x_params[2] * x[5]
            # vel[1] = non_y_params[0] * x[2] + non_y_params[1] * x[3] + non_y_params[2] * x[5]
            # ang_vel = non_theta_params[0] * x[2] + non_theta_params[1] * x[3] + non_theta_params[2] * x[5]
        return torch.cat([pos, vel, torch.atleast_1d(angle), torch.atleast_1d(ang_vel)])

    def set_params(self, tableDampingX, tableDampingY, restitution, rimFriction, tableFrictionX=None,
                   tableFrictionY=None):
        self.tableDampingX = tableDampingX
        self.tableDampingY = tableDampingY
        self.tableFrictionX = tableFrictionX
        self.tableFrictionY = tableFrictionY
        self.table.set_dynamic_parameter(tableDampingX=tableDampingX, tableDampingY=tableDampingY,
                                         rimFriction=rimFriction, restitution=restitution)

    def apply_collision(self, state, Q_collision=None, beta=1, coll_mode=False, cal_mode='spong'):
        return self.table.apply_collision(state, Q_collision=Q_collision, beta=beta, coll_mode=coll_mode,
                                          cal_mode=cal_mode)

    def close_boundary(self, state):
        return self.table.close_boundary(state)

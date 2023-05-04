import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


plot = False
save_coll = True
save_no_coll = False
if save_coll:
    coll_input = np.load('./data_linear_regression/coll_input.npy', allow_pickle=True)
    coll_output = np.load('./data_linear_regression/coll_output.npy', allow_pickle=True)
    s_slide = np.load('./data_linear_regression/s_slide_list.npy')
    s_no_slide = np.load('./data_linear_regression/s_no_slide_list.npy')
    s_list = np.concatenate((s_slide, s_no_slide))
    plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(np.vstack(coll_input)[:, 2], np.vstack(coll_input)[:, 3] * s_list,
                  np.vstack(coll_output)[:, 3], label='n slide', c='b', alpha=0.2)
    ax1.set_xlabel('b', fontsize=20)
    ax1.set_ylabel('n*s', fontsize=20)
    ax1.set_zlabel('n_out', fontsize=20)
    ax1.legend()
    plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(np.vstack(coll_input)[:, 2], np.vstack(coll_input)[:, 3],
                  np.vstack(coll_output)[:, 3], label='n slide', c='b', alpha=0.2)
    ax2.set_xlabel('b', fontsize=20)
    ax2.set_ylabel('n', fontsize=20)
    ax2.set_zlabel('n_out', fontsize=20)
    ax2.legend()
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.scatter3D(np.vstack(coll_input)[:, 2], np.vstack(coll_input)[:, 3],
                  np.vstack(coll_output)[:, 2], label='b', c='b', alpha=0.2)
    ax3.set_xlabel('b', fontsize=20)
    ax3.set_ylabel('n', fontsize=20)
    ax3.set_zlabel('b_out', fontsize=20)
    ax3.legend()
    plt.figure()
    ax4 = plt.axes(projection='3d')
    ax4.scatter3D(np.vstack(coll_input)[:, 2], np.vstack(coll_input)[:, 3] * s_list,
                  np.vstack(coll_output)[:, 2], label='b', c='b', alpha=0.2)
    ax4.set_xlabel('b', fontsize=20)
    ax4.set_ylabel('n', fontsize=20)
    ax4.set_zlabel('b_out', fontsize=20)
    ax4.legend()
    plt.show()

    b_X = coll_input[:, [2, 3, 5]]
    b_X[:, 1] = b_X[:, 1] * s_list
    b_y = coll_output[:, 2]
    reg_b = LinearRegression(fit_intercept=False).fit(b_X, b_y)
    n_X = coll_input[:, [2, 3, 5]]
    n_Y = coll_output[:, 3]
    theta_X = coll_input[:, [2, 3, 5]]
    theta_X[:, 1] = theta_X[:, 1] * s_list
    theta_Y = coll_output[:, 5]
    reg_n = LinearRegression(fit_intercept=False).fit(n_X, n_Y)
    reg_theta = LinearRegression(fit_intercept=False).fit(theta_X, theta_Y)
    b_params = np.append(reg_b.coef_, reg_b.intercept_)
    n_params = np.append(reg_n.coef_, reg_n.intercept_)
    theta_params = np.append(reg_theta.coef_, reg_theta.intercept_)
    np.save('./dyna_params/coll_params_b', b_params)
    np.save('./dyna_params/coll_params_n', n_params)
    np.save('./dyna_params/coll_params_theta', theta_params)
# plot output b  input bn btheta ntheta
if plot:
    plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter(coll_input[:, 2], coll_input[:, 3] * s_list, coll_output[:, 5], c='b', alpha=0.2)
    ax1.scatter(coll_input[:, 2], coll_input[:, 3] * s_list, reg_theta.predict(theta_X), c='r', alpha=0.2)
    ax1.set_xlabel('b', fontsize=20)
    ax1.set_ylabel('n', fontsize=20)
    ax1.set_zlabel('theta_out', fontsize=20)
    ax1.legend()
    plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.scatter(coll_input[:, 2], coll_input[:, 5], coll_output[:, 5], c='b', alpha=0.2)
    ax2.scatter(coll_input[:, 2], coll_input[:, 5], reg_theta.predict(theta_X), c='r', alpha=0.2)
    ax2.set_xlabel('b', fontsize=20)
    ax2.set_ylabel('theta', fontsize=20)
    ax2.set_zlabel('theta_out', fontsize=20)
    ax2.legend()
    plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.scatter(coll_input[:, 3] * s_list, coll_input[:, 5], coll_output[:, 5], c='b', alpha=0.2)
    ax3.scatter(coll_input[:, 3] * s_list, coll_input[:, 5], reg_theta.predict(theta_X), c='r', alpha=0.2)
    ax3.set_xlabel('n', fontsize=20)
    ax3.set_ylabel('theta', fontsize=20)
    ax3.set_zlabel('theta_out', fontsize=20)
    ax3.legend()
    plt.show()
if save_no_coll:
    non_coll_input = np.load('./data_linear_regression/non_coll_input.npy', allow_pickle=True)
    non_coll_output = np.load('./data_linear_regression/non_coll_output.npy', allow_pickle=True)
    b_X = non_coll_input[:, [2, 3, 5]]
    b_X[:, 1] = b_X[:, 1]
    b_y = non_coll_output[:, 2]
    reg_b = LinearRegression(fit_intercept=False).fit(b_X, b_y)
    n_X = non_coll_input[:, [2, 3, 5]]
    n_Y = non_coll_output[:, 3]
    theta_X = non_coll_input[:, [2, 3, 5]]
    theta_Y = non_coll_output[:, 5]
    reg_n = LinearRegression(fit_intercept=False).fit(n_X, n_Y)
    reg_theta = LinearRegression(fit_intercept=False).fit(theta_X, theta_Y)
    b_params = np.append(reg_b.coef_, reg_b.intercept_)
    n_params = np.append(reg_n.coef_, reg_n.intercept_)
    theta_params = np.append(reg_theta.coef_, reg_theta.intercept_)
    np.save('non_coll_params_b', b_params)
    np.save('non_coll_params_n', n_params)
    np.save('non_coll_params_theta', theta_params)
a = 0

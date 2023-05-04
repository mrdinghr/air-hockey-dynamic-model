# air-hockey-dynamic-model
## Description
In this repository, we build the dynamic model of puck movement in robot air hockey task. We also build Extended Kalman Filter and RTS-smoother for the puck movement.
We build two kinds of model. First is linear collision model. The tangential, normal and rotation velocity after collision is linear combination of tangential, normal and rotation velocity before collision.

Second is grey box model. The white part is based on [pong's model](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b7a85eb673dde3949baf0acb17a2dcbcb8e18435).
The black part is Neural Network. We use NN to simulate the residual dynamics of Spong's white box model.

## Folder
* folder 'data': recorded trajectories after processing
* folder 'data_linear_regression': data for calculating the parameters of linear collision model
* folder 'data_preprocess': code to deal with rosbag data
* folder 'dyna_params': parameters of linear collision model

## Code
* 'torch_air_hockey_baseline_no_detach': dynamic model of puck movement
* 'torch_EKF_wrapper': EKF and RTS Smoother model 
* 'torch_gradient': NN training process
The above three files are main part of the grey box model



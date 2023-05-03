* 'alldata': trained neural network models
* 'data': parameters for linear collision model
* 'data_linear_regression': useful data for calculating parameters of linear collision model using linear regression method
* 'data_preprocess': code for deal with rosbag data
* 'dyna_params': dynamic parameters for linear collision model

In this project, there are mainly two kinds of collision model for puck movement.
First is linear collision model. The tangential, normal and rotation velocity after collision is linear combination of tangential, normal and rotation velocity before collision.

Second is grey box model. The white part is based on [pong's model](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b7a85eb673dde3949baf0acb17a2dcbcb8e18435).
The black part is Neural Network. We use NN to simulate the residual dynamics of Spong's white box model.

* 'torch_air_hockey_baseline_no_detach': dynamic model of puck movement
* 'torch_EKF_wrapper': EKF model 
* 'torch_gradient': trainging for NN

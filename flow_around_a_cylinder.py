"""
Author: Zongren Zou
    PINNs (physical-informed neural network) for flow around a cylinder simulation
        1. Velocity field: [u, v]
        u_x + v_y = 0, input [x, y], output [u, v]
        OR
        2. Potential function: phi
        phi_xx + phi_yy = 0, input [x, y], output phi
        Notice that it is a Laplace's equation and velocity field can be derived as [u, v] = [phi_x, phi_y]

        Boundary condition:
        u = U, v = 0 when x^2 + y^2 very large
        &
        no velocity vertical to cylinder surface when x^2 + y^2 = R^2
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from preprocess import get_cylinder_data, get_outside_cylinder_data


class Model(nn.Module):
    def __init__(self, r=1, u=3, x_max=10):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 30))
        self.net.add_module('relu_layer_1', nn.ReLU())
        for i in range(2, 8):
            self.net.add_module('linear_layer_%d' %(i), nn.Linear(30, 30))
            self.net.add_module('relu_layer_%d' %(i), nn.ReLU())
        self.net.add_module('linear_layer_10', nn.Linear(30, 2))

        # boundary
        self.R = r
        self.U = u  # velocity when [x, y] is far away from the cylinder at the center
        self.end = x_max  # far away from the cylinder at the center

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        out = self.net(x)
        u, v = out[:, 0], out[:, 1]
        
        u_x, _ = torch.autograd.grad(u, x,
                                     grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
        _, v_y = torch.autograd.grad(v, x,
                                     grad_outputs=torch.ones_like(v),
                                     create_graph=True)[0]
        return ((u_x + v_y) ** 2).mean()

    def loss_bound_cylinder(self, x_cylinder):
        out = self.net(x_cylinder)
        u, v = out[:, 0], out[:, 1]
        sin_theta = x_cylinder[:, 1] / self.R
        cos_theta = x_cylinder[:, 0] / self.R
        return ((u * cos_theta + v * sin_theta) ** 2).mean()

    def loss_bound_far_away(self, x_far_away):
        out = self.net(x_far_away)
        u, v = out[:, 0], out[:, 1]
        return ((u-self.U) ** 2 + v ** 2).mean()

def main():
    ## parameters
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = 1000
    lr = 0.001
    
    ## boundary sampling points, inner sampling points, initial sampling points
    num_x, num_y, num_cylinder = 100, 100, 30
    x_max, y_max = 20, 8
    r = 1
    u = 10
    # load data
    x_inner, x_far_away = get_outside_cylinder_data(num_x, num_y, r, x_max, y_max)
    x_cylinder = get_cylinder_data(num_cylinder, r)
    # transfer data to the device
    x_inner_tensor = torch.tensor(x_inner, requires_grad=True, dtype=torch.float32).to(device)
    x_far_away_tensor = torch.tensor(x_far_away, requires_grad=True, dtype=torch.float32).to(device)
    x_cylinder_tensor = torch.tensor(x_cylinder, requires_grad=True, dtype=torch.float32).to(device)

    ## instantiate model and optimizer
    model = Model(r=r, u=u, x_max=x_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_inner_tensor)
            loss_bound_cylinder = model.loss_bound_cylinder(x_cylinder_tensor)
            loss_bound_far_away = model.loss_bound_far_away(x_far_away_tensor)

            loss = loss_pde + loss_bound_cylinder + loss_bound_far_away
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}')
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

if __name__ == '__main__':
    main()



















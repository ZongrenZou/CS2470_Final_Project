"""
Author: Zongren Zou
    Pre-process data for flow around a cylinder simulation
    Specifically, generate x_train, x_cylinder, x_far_away, assume the domain is very large plane with a disk at the center
"""

import numpy as np


def get_outside_cylinder_data(num_x, num_y, r, x_max, y_max):
    x = np.linspace(-x_max, x_max, num_x).reshape([num_x, 1])
    y = np.linspace(-y_max, y_max, num_y).reshape([num_y, 1])
    x, y = np.meshgrid(x, y)
    points = np.array([x.flatten(), y.flatten()]).T  # shape of [num_x*num_y, 2]
    outside_cylinder = points[:, 0] ** 2 + points[:, 1] ** 2 >= r ** 2
    outside_points = points[outside_cylinder, :]
    far_away = np.abs(outside_points[:, 0]) == x_max
    return outside_points[np.invert(far_away), :], outside_points[far_away, :]


def get_cylinder_data(num, r):
    theta = np.linspace(0, 2*np.pi, num)
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return np.array([x, y]).T
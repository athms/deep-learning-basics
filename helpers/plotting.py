#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_gradient_descent_path(perceptron, X, y):
    """This is a helper function for 
    the exercise of 0-Perceptron-Gradient-Descent

    Args:
        perceptron (class): perceptron implementation
        X (ndarray): the data (instaces x features)
        y (array): labels for each data instance
    """
    
    # extract the weights for each gradient step
    training_w = np.array(perceptron.training_w)
    b = float(perceptron.b) # also extract bias
    
    # define steps will be plotted
    n_steps = len(training_w)
    suggested_steps = np.array([0, 1, 2, 10, 50, 5000])
    steps = np.array([s for s in suggested_steps if s < n_steps])
    steps = np.append(steps, -1).astype(np.int)

    # compute the values of the loss function for a grid of w-values, given our learned bias term
    w1_vals = np.linspace(np.min([np.min(training_w[:,0]), -10]),
                          np.max([np.max(training_w[:,0]), 30]), 100)
    w2_vals = np.linspace(np.min([np.min(training_w[:,1]), -10]),
                          np.max([np.max(training_w[:,1]), 30]), 100)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    grid_r, grid_c = W1.shape
    ZZ = np.zeros((grid_r, grid_c))
    for i in range(grid_r):
        for j in range(grid_c):
            w = np.array([W1[i,j], W2[i,j]])
            y_pred = perceptron.activation(X.dot(w)+b)
            ZZ[i, j] += np.nan_to_num(np.mean(perceptron.loss(y, y_pred)))


    # plot the loss function and gradient descent steps
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    # plot contour
    cs = ax.contourf(W1, W2, ZZ, 50, vmax=10, cmap=cm.viridis)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    # plot trajectory
    ax.scatter(training_w[steps,0], training_w[steps,1], color='white')
    # mark start point
    ax.scatter(training_w[0,0], training_w[0,1], color='black', s=200, zorder=99)
    ax.plot(training_w[steps,0], training_w[steps,1], color='white', lw=1)
    # add line for final weights
    ax.axvline(training_w[-1,0], color='red', lw=1, ls='--')
    ax.axhline(training_w[-1,1], color='red', lw=1, ls='--')
    # label axes
    cbar.set_label('Loss')
    ax.set_title('Final loss: {}'.format(perceptron.training_loss[-1]))
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    
    return fig, ax
from __future__ import division

import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np

'''
This is support code provided for the Bayesian Regression Problems.
The goal of this problem is to have you explore Bayesian Linear Gaussian Regression, as described in Lecture.
In particular, the goal is to reproduce fig 3.7 from Bishop's book.

A few things to note about this code:
    - We strongly encourage you to review this support code prior to
      completing "problem.py"
    - For Problem (b), you are asked to generate plots for three
      values of sigma_squared. We suggest you save the plot generated
      by make_plots (instead of simply calling plt.show)
'''

def generate_data(data_size, noise_params, actual_weights):
    # x1: from [0,1) to [-1,1)
    x1 = -1 + 2 * np.random.rand(data_size, 1)
    # appending the bias term
    xtrain = np.matrix(np.c_[np.ones((data_size, 1)), x1])
    # random noise
    noise = np.matrix(np.random.normal(
                      noise_params["mean"],
                      noise_params["var"],
                      (data_size, 1)))

    ytrain = (xtrain * actual_weights) + noise

    return xtrain, ytrain

def make_plots(actual_weights, xtrain, ytrain, likelihood_var, prior, likelihood_func, get_posterior_params, get_predictive_params):

    # #setup for plotting
    #
    show_progress_till_data_rows = [1, 2, 10, -1]
    num_rows = 1 + len(show_progress_till_data_rows)
    num_cols = 4
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=.8, wspace=.8)

    plot_without_seeing_data(prior, num_rows, num_cols)

    # see data for as many rounds as specified and plot
    for round_num, row_num in enumerate(show_progress_till_data_rows):
        current_row = round_num + 1
        first_column_pos = (current_row * num_cols) + 1

        # #plot likelihood on latest point
        plt.subplot(num_rows, num_cols, first_column_pos)


        likelihood_func_with_data = lambda W: likelihood_func(W,
                                                      xtrain[:row_num,],
                                                      ytrain[:row_num],
                                                      likelihood_var)
        contour_plot(likelihood_func_with_data, actual_weights)

        # plot updated posterior on points seen till now
        x_seen = xtrain[:row_num]
        y_seen = ytrain[:row_num]
        mu, cov = get_posterior_params(x_seen, y_seen,
                                      prior, likelihood_var)
        posterior_distr = multivariate_normal(mu.T.tolist()[0], cov)
        posterior_func = lambda x: posterior_distr.pdf(x)
        plt.subplot(num_rows, num_cols, first_column_pos + 1)
        contour_plot(posterior_func, actual_weights)

        # plot lines
        data_seen = np.c_[x_seen[:, 1], y_seen]
        plt.subplot(num_rows, num_cols, first_column_pos + 2)
        plot_sample_lines(mu, cov, data_points=data_seen)

        # plot predictive
        plt.subplot(num_rows, num_cols, first_column_pos + 3)
        post_mean, post_var = get_posterior_params(x_seen, y_seen, prior)
        plot_predictive_distribution(get_predictive_params, post_mean, post_var)

    # #show the final plot
    plt.show()

def plot_without_seeing_data(prior, num_rows, num_cols):

    #Blank likelihood
    plt.subplot(num_rows, num_cols, 1, facecolor='grey')
    plt.title("Likelihood")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.9, 0.9])
    plt.ylim([-0.9, 0.9])

    #Prior
    prior_distribution = multivariate_normal(mean=prior["mean"].T.tolist()[0],
        cov=prior["var"])
    prior_func = lambda x:prior_distribution.pdf(x)
    plt.subplot(num_rows, num_cols, 2)
    plt.title("Prior/Posterior")
    contour_plot(prior_func)

    # Plot initially valid lines (no data seen)
    plt.subplot(num_rows, num_cols, 3)
    plt.title("Data Space")
    plot_sample_lines(prior["mean"], prior["var"])

    # Blank predictive
    plt.subplot(num_rows, num_cols, 4, facecolor='grey')
    plt.title('Predictive Distribution')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("")
    plt.ylabel("")

def contour_plot(distribution_func, actual_weights=[]):

    step_size = 0.05
    array = np.arange(-1, 1, step_size)
    x, y_train = np.meshgrid(array, array)

    length = x.shape[0] * x.shape[1]
    x_flat = x.reshape((length, 1))
    y_flat = y_train.reshape((length, 1))
    contour_points = np.c_[x_flat, y_flat]

    values = list(map(distribution_func, contour_points))
    values = np.array(values).reshape(x.shape)

    plt.contourf(x, y_train, values)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.xticks([-0.5, 0, 0.5])
    plt.yticks([-0.5, 0, 0.5])
    plt.xlim([-0.9, 0.9])
    plt.ylim([-0.9, 0.9])

    if(len(actual_weights) == 2):
        plt.plot(float(actual_weights[0]), float(actual_weights[1]),
                 "*k", ms=5)

# Plot the specified number of lines of the form y_train = w0 + w1*x in [-1,1]x[-1,1] by
# drawing w0, w1 from a bivariate normal distribution with specified values
# for mu = mean and sigma = covariance Matrix. Also plot the data points as
# circles.
def plot_sample_lines(mean, variance,
                    number_of_lines=6,
                    data_points=np.empty((0, 0))):
    step_size = 0.05
    # generate and plot lines
    for round in range(1, number_of_lines):
        weights = np.matrix(np.random.multivariate_normal(mean.T.tolist()[0], variance)).T
        x1 = np.arange(-1, 1, step_size)
        x = np.matrix(np.c_[np.ones((len(x1), 1)), x1])
        y_train = x * weights

        plt.plot(x1, y_train)

    # markings
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")

    # plot data points if given
    if(data_points.size):
        plt.plot(data_points[:, 0], data_points[:, 1],
                 "co")

def plot_predictive_distribution(get_predictive_params,post_mean, post_var):
    step_size = 0.05
    x = np.arange(-1, 1, step_size)
    x = np.matrix(np.c_[np.ones((len(x), 1)), x])
    pred_means = np.zeros(x.shape[0])
    pred_stds = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        pred_means[i], pred_stds[i] = get_predictive_params(x[i,].T,
                                                        post_mean,
                                                        post_var)
    pred_stds = np.sqrt(pred_stds)
    plt.plot(x[:,1], pred_means, 'b')
    plt.plot(x[:,1], pred_means + pred_stds, 'b--')
    plt.plot(x[:,1], pred_means - pred_stds, 'b--')
    plt.xticks([-1, 0, 1])
    plt.yticks([-0.5, 0, 0.5])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")

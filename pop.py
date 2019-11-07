# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:52:21 2018

@author: 27279
"""

#from kf_book.book_plots import figsize
#import matplotlib.pyplot as plt
#import kf_book.gh_internal as gh
#import kf_book.book_plots as book_plots
#from kf_book.book_plots import plot_errorbars
#import book_format
#book_format.set_style()
#
#weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 
#           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
#
#time_step = 1.0  # day
#scale_factor = 4.0/10
#
#def predict_using_gain_guess(weight, gain_rate, do_print=False):     
#    # store the filtered results
#    # 存储过滤结果
#    estimates, predictions = [weight], []
#
#    # most filter literature uses 'z' for measurements
#    # 大多数滤波器文献使用'z'进行测量
#    for z in weights: 
#        # predict new position--预测新的位置
#        prediction = weight + gain_rate * time_step
#
#        # update filter 
#        weight = prediction + scale_factor * (z - prediction)
#
#        # save
#        estimates.append(weight)
#        predictions.append(prediction)
#        if do_print:
#            gh.print_results(estimates, prediction, weight)
#
#    return estimates, predictions  #估计、 预测、 以前(previous)
#
#initial_guess = 160.
#
#estimates, predictions = predict_using_gain_guess(
#    weight=initial_guess, gain_rate=1, do_print=True)


import random
from scipy.stats import norm
import matplotlib.pyplot as plt

def cauchy(theta):
    y = 1.0 / (1.0 + theta ** 2)
    return y

T = 5000
sigma = 1        #方差
thetamin = -30
thetamax = 30
theta = [0.0] * (T+1)
theta[0] = random.uniform(thetamin, thetamax)

t = 0
while t < T:
    t = t + 1
    theta_star = norm.rvs(loc=theta[t - 1], scale=sigma, size=1, random_state=None)
    #print theta_star
    a = theta_star[0]
    b = theta[t - 1]
    alpha = min(1, (cauchy(a) / cauchy(b)))

    u = random.uniform(0, 1)
    if u <= alpha:
        theta[t] = theta_star[0]
    else:
        theta[t] = theta[t - 1]

ax1 = plt.subplot(211)
ax2 = plt.subplot(212) 
plt.sca(ax1)
plt.ylim(thetamin, thetamax)
plt.plot(range(T+1), theta, 'g-')
plt.sca(ax2)
num_bins = 50
plt.hist(theta, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.show()




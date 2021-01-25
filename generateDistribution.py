from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 16
LARGE_SIZE = 18

plt.rc('font', size=SMALL_SIZE)             # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)        # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)      # fontsize of the figure title

price_list_g = [4.5, 7, 15]
# mu_g = 0.0
# sigma_g = 1.0
# dis_m50e25
mu_g = 7
sigma_g = 11.86
# dis_m99e96
# mu_g = 38
# sigma_g = 13.14

X_g = np.arange(0, 20, 0.001)
Y_g = [stats.norm.pdf(X_g, mu_g, sigma_g), stats.norm.cdf(X_g, mu_g, sigma_g), stats.norm.sf(X_g, mu_g, sigma_g)]

# -- plot --
X_label = ['Prediction of Purchasing Ability', 'Prediction of Purchasing Ability', 'Prediction of Purchasing Ability']
Y_label = ['Probability Density', 'Cumulative Probability', 'Cumulative Probability']
title = ['PDF', 'CDF', 'dis_m50e25']

for index in range(3):
    plt.plot(X_g, Y_g[index])
    if index == 0:
        for pk in range(len(price_list_g)):
            plt.plot(price_list_g[pk], float(stats.norm.pdf(price_list_g[pk], mu_g, sigma_g)), '*')
            plt.text(price_list_g[pk], float(stats.norm.pdf(price_list_g[pk], mu_g, sigma_g)),
                     round(float(stats.norm.pdf(price_list_g[pk], mu_g, sigma_g)), 4), ha='center', va='bottom')
    elif index == 1:
        for pk in range(len(price_list_g)):
            plt.plot(price_list_g[pk], float(stats.norm.cdf(price_list_g[pk], mu_g, sigma_g)), '*')
            plt.text(price_list_g[pk], float(stats.norm.cdf(price_list_g[pk], mu_g, sigma_g)),
                     round(float(stats.norm.cdf(price_list_g[pk], mu_g, sigma_g)), 4), ha='center', va='bottom')
    elif index == 2:
        for pk in range(len(price_list_g)):
            plt.plot(price_list_g[pk], float(stats.norm.sf(price_list_g[pk], mu_g, sigma_g)), '*')
            plt.text(price_list_g[pk], float(stats.norm.sf(price_list_g[pk], mu_g, sigma_g)),
                     round(float(stats.norm.sf(price_list_g[pk], mu_g, sigma_g)), 4), ha='center', va='bottom')

    plt.xlabel(X_label[index])
    plt.ylabel(Y_label[index])
    plt.title(title[index])
    # plt.title(title[index] + ': μ = ' + str(round(mu_g, 4)) + ', σ = ' + str(round(sigma_g, 2)))
    plt.grid()
    # save_name = 'distribution/' + title[index]
    # plt.savefig(save_name)
    plt.show()
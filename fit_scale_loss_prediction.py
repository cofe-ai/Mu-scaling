from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt 

# model widths
shapes = [128, 256, 384, 512, 640, 768,  896, 1024, 2048, 3072]
# number of parameters in millions
x = [8.53, 21.56, 39.09, 61.12, 87.65, 118.68, 154.21, 194.24, 676.48, 1446.72] 
# training loss 
train_loss = [3.9181,3.6054,3.4431,3.3542,3.2902,3.2478,3.224,3.1797,3.0854, 3.0425]

y = np.array(train_loss)
x_sample = range(int(min(x)) - 1,int(max(x)) + 5,1)

# power law for fitting
def func(x, a, b,c):
    return a * np.power(x, b) + c

def curve_fit_one_line(x, y, num_pred, x_sample):
    popt, pcov = curve_fit(func, x[:10-num_pred], y[:10-num_pred], p0=[1, -1, 3], maxfev=5000)
    a = popt[0] 
    b = popt[1]
    c = popt[2]
    print(a, b, c)
    print(np.sqrt(np.diag(pcov)))

    yvals = func(x_sample, a, b, c)
    return popt, np.sqrt(np.diag(pcov)), yvals

# Number of models used for prediction. Results for prediction are not used in fitting curves
num_pred=3
popt, perr, yvals = curve_fit_one_line(x, y, num_pred, x_sample)

plot1 = plt.scatter(x[:10-num_pred], y[:10-num_pred], s=15, c='c')
plot1 = plt.scatter(x[10-num_pred:], y[10-num_pred:], s=30, c='c', marker="*")
plot2 = plt.plot(x_sample, yvals, 'c', ls="--", label='(7.5e-4, 0.04, 6.0)')

for _x, _y, _s in zip(x, y, shapes):
    plt.text(_x * 0.99, _y * 0.99, f"{_s}", horizontalalignment='right', verticalalignment='top')
plt.legend()
#plt.xscale("log")
plt.xlabel("model size / M")
plt.ylabel("train_loss @ 20k")
plt.savefig(f"train_loss_prediction_20k_check.png")



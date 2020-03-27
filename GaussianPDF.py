# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:50:27 2020

Gaussian Probability Distribution Fitting

the red and blue histograms are aspect ratio hist before and after smoothing
the red and blue curves are Gaussian PDF before and after smoothing

@author: Wen Wen
"""
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv

# Generate some data for this demonstration.
#data = norm.rvs(10.0, 2.5, size=500)

path='D:/Private Manager/Personal File/uOttawa/Thesis/figures/dataset analysis/lowpass_and_Gaussian/'
#filename='bdd_hist_lp_abs.csv'
filename=['bdd_hist_r_cut.csv','bdd_hist_lp_abs.csv']
histcolor=['r', 'b']

plt.figure(figsize=(9,6),dpi=100)

for file, color in zip(filename,histcolor):
    histdata = []
    with open(os.path.join(path,file), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            histdata.append(row)
    histdata=[int(float(i)*100) for i in histdata[0]]
    
    data=np.array([])
    for d, i  in zip(histdata,range(len(histdata))):
        value=float(i)/800.0*3.0
        data=np.concatenate((data, np.array(d*[value])),axis=0)
    
    # Fit a normal distribution to the data:
    mu = np.mean(data)
    std = np.std(data)
    
    # Plot the histogram.
    plt.hist(data, bins=200,normed=True, alpha=0.4, color=color)


# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
p_before_lp = norm.pdf(x, 1.31, 0.66)
plt.plot(x, p, 'b', linewidth=2)
plt.plot(x, p_before_lp, 'r', linewidth=2)
title = "Fit results: mean = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.savefig(os.path.join(path,'GaussianPDF.png'))

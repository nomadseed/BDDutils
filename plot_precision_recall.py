# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:35:09 2019

plot precision vs recall curve

@author: Wen Wen
"""
import json
import os
import matplotlib.pyplot as plt

folder_list=['s4cnn',
             'ssd_mobilenet_opt_150_gt22',
             'ssd_mobilenet_opt_300_gt22']
threshlist=[0.0001,
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
            0.9999] #21 numbers in total

def plotPRcurve(performance, savepath=None):
    chartaxis = [0.0,1.0,0.0,1.0]
    plt.figure(figsize=(8,6),dpi=100)
    plt.axis(chartaxis)
    
    for model in performance:
        precision=[]
        recall=[]
        for conf in performance[model]:
            precision.append(performance[model][conf]['overall']['precision'])
            recall.append(performance[model][conf]['overall']['recall'])
        plt.plot(recall,precision,'o-',label=model)
    plt.title('Precision vs Recall')
    plt.legend(loc=3)

    if savepath is not None:
        plt.savefig(os.path.join(savepath,'precision_vs_recall.png'))
    plt.show()
    
def loadjson(loadpath,folderlist=folder_list):
    performance={}
    
    for folder in folderlist:
        
        with open(os.path.join(loadpath,folder,'IOU 0_5','performance.json')) as fp:
            performance[folder]=json.load(fp)
    
    return performance

if __name__=='__main__':
    loadpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/detection results'
    
    performance=loadjson(loadpath)
    plotPRcurve(performance,savepath=loadpath)
    
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:35:09 2019

plot precision vs recall curve

@author: Wen Wen
"""
import json
import os
import matplotlib.pyplot as plt

folder_list={'ssd_mobilenet_v2_300_bdd_gt22':'Original SSD 300',
             'ssd_mobilenet_v2_224_bdd_autoratio_gt22':'SSD 224 (anchor ratio changed)',
             #'s4cnn':'S4CNN',
             #'s4cnn_gt22':'S4CNN',
             'ssd_mobilenet_v2_300_bdd_autoratio_gt22':'SSD 300 (anchor ratio changed)',
             #'ssd_mobilenet_opt_300':'Our SSD 300 (belt-shape grouping)',
             'ssd_mobilenet_opt_300_clust23p_gt22':'SSD 300 (simple 23 pts K-means)',
             'ssd_mobilenet_opt_300_gt22':'Our SSD 300 (belt-shape grouping)',
             'ssd_mobilenet_opt_300_clust2_gt22':'Our SSD 300 (sector-shape grouping)'
             }
legendlist={'Original SSD 300':[0.4745098, 0.9372549, 1.       ],
            'SSD 224 (anchor ratio changed)':[0.21176471, 0.85098039, 0.98039216],
            'S4CNN':[0.2, 1., 0.],
            'SSD 300 (anchor ratio changed)':[0.12941176, 0.66666667, 1.        ],
            'SSD 300 (simple 23 pts K-means)':[0, 0.933, 0.1],
            'Our SSD 300 (belt-shape grouping)':[1. , 0. , 0.4],
            'Our SSD 300 (sector-shape grouping)':[1.        , 0.75294118, 0.        ]
            }
threshlist=[0.0001, 0.01, 0.02, 0.03, 0.04, 
                0.05, 0.1, 0.15, 0.2, 0.25, 
                0.3, 0.35, 0.4, 0.45, 0.5, 
                0.55, 0.6, 0.65, 0.7, 0.75, 
                0.8, 0.85, 0.9, 0.95, 0.99] #25 numbers in total

def plotPRcurve(performance, savepath=None):
    chartaxis = [0.0,1.0,0.0,1.0]
    
    categorylist=['overall','small','medium','large']
    legendloc={'overall':3,
               'small':1,
               'medium':3,
               'large':3
               }
    
    for category in categorylist:
        plt.figure(figsize=(8,6),dpi=100)
        plt.axis(chartaxis)
        for model in performance:
            precision=[]
            recall=[]
            for conf in performance[model]:
                precision.append(performance[model][conf][category]['precision'])
                recall.append(performance[model][conf][category]['recall'])
            plt.plot(recall,precision,marker='o',color=legendlist[model],label=model)
            
            # calculate AP from precision and recall
            ap=getAP(recall,precision)
            print('model:{} cate:{} AP={}'.format(model,category,ap))
            
        plt.title(' Precision vs Recall ({})'.format(category))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc=legendloc[category])
    
        if savepath is not None:
            plt.savefig(os.path.join(savepath,category+'_precision_vs_recall.png'))
        plt.show()
    return recall, precision

def getAP(recall, precision, _type='left'):
    if len(recall)!= len(precision):
        raise ValueError("precision and recall list should have same length !")
    if _type=='left':
        last_x=0
        ap=0
        recall.reverse()
        precision.reverse()
        for x, y in zip(recall,precision):
            ap+=(x-last_x)*y
            last_x=x
        return ap
   
def loadjson(loadpath,folderlist=folder_list):
    performance={}
    
    for folder in folderlist:
        
        with open(os.path.join(loadpath,folder,'IOU 0_5 3296','performance.json')) as fp:
            performance[folderlist[folder]]=json.load(fp)
    
    return performance

if __name__=='__main__':
    loadpath='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/detection results'
    
    performance = loadjson(loadpath)
    recall, precision = plotPRcurve(performance,savepath=loadpath)
    
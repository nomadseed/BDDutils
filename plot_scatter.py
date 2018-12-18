# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:24:29 2018

plot annotation anchors distribution and k-mean result

@author: Wen Wen
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

class Plot_scatter(object):
    """
    to plot scatter chart for showing k-mean result and anchor width and height
    distribution
    
    """
    def __init__(self):
        self.savepath=''
        self.boxlist=[]
        self.anchorlist=[]
        self.config={'title':'',
                     'legend':[]}
        self.figure=None

    def set_save_path(self,path):
        self.savepath=path
    
    def set_config(self, config):
        configsetting={'title','legend'}
        if config is not dict:
            assert TypeError('the config file has to be a dictionary')        
        for i in config:
            if i not in configsetting:
                assert ValueError('the specified config key is not in the setting')
            self.config[i]=config[i]
        
    def load_sample_data(self):
        self.boxlist=[[25,25],[50,50],[30,40],[45,45]]
        self.anchorlist=[[25,25],[41.6,45]]
    
    def load_box_list(self,boxlist):
        if len(boxlist)==0:
            assert ValueError('empty boxlist')
        if len(boxlist[0])!=2:
            assert ValueError('the dimension of box list shoule be a Nx2 array')
        self.boxlist=boxlist
        
    def append_box(self,singlebox):
        self.boxlist.append(singlebox)
        
    def get_box_list(self):
        return self.boxlist
        
    def load_anchor_list(self,anchorlist):
        self.anchorlist=anchorlist
        
    def append_anchor(self,singleanchor):
        self.anchorlist.append(singleanchor)
    
    def get_anchor_list(self):
        return self.anchorlist
    
    def is_ready(self):
        if self.boxlist==[]:
            return False
        return True
    
    def save_scatter(self):
        x = [i[0] for i in self.boxlist]
        y = [i[1] for i in self.boxlist]
        area = np.ones(len(self.boxlist))*1
        ax = plt.subplot()
        ax.scatter(x, y, s=area, alpha=0.3, c='b', marker='o')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        plt.grid(True)
        plt.savefig(os.path.join(self.savepath,'scatter.png'),dpi=1000)
        print('scatter saved as\n {}'.format(os.path.join(self.savepath,'scatter.png')))
        plt.show()
        plt.close()
        
    
def load_annotation(jsonpath):
    if not os.path.exists(jsonpath):
        print('try loading "{}"'.format(jsonpath))
        print('no such a json file, an empty list will be returned')
        return []
    
    annodict=json.load(open(jsonpath))
    boxlist=[]
    for imgname in annodict:
        for box in annodict[imgname]['annotations']:
            boxlist.append([box['width'],box['height']])
    return boxlist
        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/labels/bdd100k_labels_images_val_VIVA_format_crop.json', 
                        help="select the path for json file")
    parser.add_argument('--clust_path', type=str, 
                        default='clustpath', 
                        help="select the path for clustered result")
    parser.add_argument('--save_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/labels', 
                        help="path to save the figure")
    args=parser.parse_args()
    jsonpath=args.json_path
    savepath=args.save_path
    clustpath=args.clust_path
    
    
    plotter=Plot_scatter()
    
    # load json file
    #plotter.load_sample_data() #for debugging
    
    boxlist=load_annotation(jsonpath)
    anchorlist=load_annotation(clustpath)
    plotter.load_anchor_list(anchorlist)
    plotter.load_box_list(boxlist)
    
    if plotter.is_ready():
        plotter.set_save_path(savepath)
        plotter.save_scatter()
    else:
        print('plotter not ready, please double check the config and data')
    
    
        
        
        
        
""" End of File """        
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:21:58 2019

clustering for getting optimized anchors

methods:
    k-means

@author: Wen Wen
"""
from sklearn.cluster import KMeans
import numpy as np
import time
import os
import json
import argparse
import math


def load_json_annotations(filepath, jsonlabel):
    annotationdict={}
    folderdict=os.listdir(filepath)
    for foldername in folderdict:
        jsonpath=os.path.join(filepath,foldername)
        if '.' in jsonpath:
            continue
        # load the json files
        jsondict=os.listdir(jsonpath)
        for jsonname in jsondict:
            if jsonlabel in jsonname and '.json'in jsonname:
                annotationdict[foldername]={}
                annotationdict[foldername]=json.load(open(os.path.join(jsonpath,jsonname)))
    
    return annotationdict

def returnBoxArea(box):
    # return area of box
    return box[0]**2+box[1]**2

def get_bboxes(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    bboxlist=[]
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            if len(annotationdict[foldername][imagename])==0:
                continue
            else:
                for anno in annotationdict[foldername][imagename]['annotations']:
                    if anno['width'] and anno['height']:
                        bboxlist.append([anno['width'],anno['height']])
                
    return bboxlist

def groupBBox(bboxlist,threshlist):
    """
    group bbox into different groups by grid size, save all the bbox into a dict
    that contain multiple bbox lists
    
    """
    xlist=bboxlist.copy()
    xlist.sort(key=returnBoxArea,reverse=True)
    bboxdict={}
    print('grouping bbox based on threshlist...')
    for thresh in threshlist:
        curlist=[]
        for i in range(len(xlist)):
            box=xlist.pop()
            if box[0]**2+box[1]**2<thresh**2:
                curlist.append(box)
            else:
                break
    
        bboxdict[thresh]=curlist
    return bboxdict

def simpleKMeans(xlist,n_cluster=2,random_state=0,algorithm='auto'):
    """
    
    output:
        centroids: centroids acquired from clustering
        labels: clustering result of input array
        score: sum of distances from centroids to all the points in their cluster
        time: processing time of current k-means clustering
        
    """
    starttime=time.time()
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state,algorithm=algorithm).fit(xlist)
    processtime = time.time()-starttime
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = -kmeans.score(xlist)
    
    return centroids, labels, score, processtime

def adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.8,plotfig=True):
    """
    take in a bboxlist and perform k-means clustering where k ranges from 2 to
    a maximum (defalut is 10). 
    Using Elbow method to get the optimized k value for the bboxlist, where the
    elbow point is defined by a score threshold. when current score is larger
    than last score, i.e. score>scorethresh*last_score, the last point is the
    elbow point. note that the changing threshold will have huge impact to the 
    final result.
    
    """
    clusterlist=[]
    for thresh in bboxdict:
        grouplist=[]
        for n_cluster in range(2,max_num_center):
            # k-means clustering        
            centroids, labels, score, processtime = simpleKMeans(xlist=bboxdict[thresh], n_cluster=n_cluster)
            print('clustering for k={}, processing time={}, score={}'.format(n_cluster,processtime,score))
            singledict={}
            singledict['k']=n_cluster
            singledict['centroids']=centroids
            singledict['labels']=labels
            singledict['score']=score
            singledict['processtime']=processtime
            grouplist.append(singledict)
        
        clusterlist.append({'thresh':thresh, 'value':grouplist})
    return clusterlist
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k', 
                        help="File path of input data")
    parser.add_argument('--json_label', type=str, 
                        default='val_VIVA_format_crop_gt22.json', 
                        help="label to specify json file")
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get bboxlist from annotation
    bboxlist = get_bboxes(annotationdict)
    
    # setup threshlist, where threshold is diagnal length in pixel
    # to get the threshlist easily, check the get_anchor_from_bench_mark.py
    # run the file and check h_pel_list
    threshlist = [43,80,160,267,400,800]
    bboxdict = groupBBox(bboxlist,threshlist)
    
    # perform k-means for all the layers
    clusterdict = adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.75,plotfig=True)

    # draw figures of [sum of error over k values]
    




"""End of file"""
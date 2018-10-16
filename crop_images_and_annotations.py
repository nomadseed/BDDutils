# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:53:49 2018

crop BDD images and annotations into viewnyx format

@author: Wen Wen
"""

import os
import json
import argparse
import cv2

# for debug only
from matplotlib import pyplot as plt

def crop_single_image(imgname,ori_size=(720,1280),resized=(480,640)):
    """
    crop a single image from original size
    
    """
    img=cv2.imread(imgname) # note the cv2 img has [y,x,depth] format
    if img.shape[0]!=ori_size[0] or img.shape[1]!=ori_size[1]:
        raise ValueError("input image shape is not correct")
    
    resizeratio=min(ori_size[0]/resized[0],ori_size[1]/resized[1])
    h=int(resized[0]*resizeratio)
    w=int(resized[1]*resizeratio)
    
    newimg=img[round((ori_size[0]-h)*0.5):round((ori_size[0]+h)*0.5),
               round((ori_size[1]-w)*0.5):round((ori_size[1]+w)*0.5)]
    newimg=cv2.resize(newimg, (resized[1],resized[0])) 
    return newimg


def crop_images(imagepath,folder='cropped'):
    """
    crop the images in BDD dataset from 1280*720 (aspect ratio 1.777) to
    960*720 (aspect ratio 1.333), which is the same as viewnyx has (640*480, 1.333)
    
    """
    
    folderlist_k=os.listdir(imagepath) # list for 10k and 100k
    for folder_k in folderlist_k:
        folderlist_tvt=os.listdir(os.path.join(imagepath,folder_k)) # train val test
        for folder_tvt in folderlist_tvt:
            currentpath=os.path.join(imagepath,folder_k,folder_tvt)
            print('cropping images under folder '+currentpath+' \n')
            imglist=os.listdir(currentpath)
            if not os.path.exists(os.path.join(currentpath,folder)):
                os.mkdir(os.path.join(currentpath,folder))
            for imgname in imglist:
                if 'crop' in imgname:
                    continue
                img=crop_single_image(os.path.join(currentpath,imgname))
                cv2.imwrite(os.path.join(currentpath,folder,imgname.replace('.jpg','_crop.jpg')),img)
            
    print('done cropping images')            

def crop_annos(annopath):
    return 0

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k', 
        help="select the file path for BDD image and annotations")
    # debug path:
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k
    # actual path:
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    args = parser.parse_args()
    filepath=args.file_path
    imagepath=os.path.join(filepath,'images')
    annopath=os.path.join(filepath,'labels')
    
    #crop_images(imagepath)
    crop_annos(annopath)
    





""" End of file """
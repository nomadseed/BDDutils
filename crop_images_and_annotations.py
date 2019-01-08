# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:53:49 2018

crop BDD images and annotations into viewnyx format

file_path: path of the BDD images and labels
crop_image: choose to crop images or not, default as true
crop_anno: choose to crop annotations or not, default as true

@author: Wen Wen
"""

import os
import json
import argparse
import cv2

# for debug only
#from matplotlib import pyplot as plt

def crop_single_image(imgname,ori_size=(720,1280),resized=(480,640)):
    """
    crop a single image from original size
    keep the part in the middle
    """
    img=cv2.imread(imgname) # note the cv2 img has [y,x,depth] format
    if img.shape[0]!=ori_size[0] or img.shape[1]!=ori_size[1]:
        print("input image shape of {} is not correct\n".format(imgname))
        return None
    
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
                if img is not None:
                    cv2.imwrite(os.path.join(currentpath,folder,imgname.replace('.jpg','_crop.jpg')),img)
            
    print('done cropping images')            

def correct_single_box(bbx,ori_size=(720,1280),resized=(480,640)):
    newbox={}
    
    resizeratio=min(ori_size[0]/resized[0],ori_size[1]/resized[1])
    h=int(resized[0]*resizeratio)
    w=int(resized[1]*resizeratio)
    
    # remove the bbx if it's out of the cropped image
    if bbx['x']>=round((ori_size[1]+w)*0.5) or bbx['y']>=round((ori_size[0]+h)*0.5):
        return None
    elif bbx['x']+bbx['width']<=round((ori_size[1]-w)*0.5) or bbx['y']+bbx['height']<=round((ori_size[0]-h)*0.5):
        return None
    # if the bbx is partially inside of the cropped image
    else:
        x=round((bbx['x']-round((ori_size[1]-w)*0.5))/resizeratio)
        if x<0:
            x=0
        newbox['x']=x
        
        y=round((bbx['y']-round((ori_size[0]-h)*0.5))/resizeratio)
        if y<0:
            y=0
        newbox['y']=y
        
        width=round(bbx['width']/resizeratio)
        if width>resized[1]-x:
            width=resized[1]-x
        newbox['width']=width
        
        height=round(bbx['height']/resizeratio)
        if height>resized[0]-y:
            height=resized[0]-y
        newbox['height']=height
        
        newbox['id']=bbx['id']
        newbox['label']=bbx['label']
        newbox['shape']=bbx['shape']
        newbox['category']=bbx['category']
        
        return newbox

def crop_single_anno(annodict,ori_size=(720,1280),resized=(480,640),rejectsize=0):
    newdict={}
    for imgname in annodict:
        if annodict[imgname]['width']!=ori_size[1] or annodict[imgname]['height']!=ori_size[0]:
            continue
        singleimg={}
        singleimg['height']=resized[0]
        singleimg['width']=resized[1]
        singleimg['name']=annodict[imgname]['name'].replace('.jpg','_crop.jpg')
        singleimg['annotations']=[]
        for bbx in annodict[imgname]['annotations']:
            newbbx=correct_single_box(bbx,ori_size,resized)
            if newbbx!=None and newbbx['width']>rejectsize and newbbx['height']>rejectsize:
                singleimg['annotations'].append(newbbx)
        newdict[imgname.replace('.jpg','_crop.jpg')]=singleimg
    return newdict

def crop_annos(annopath,ori_size=(720,1280),resized=(480,640),rejectsize=0):
    """
    crop and correct annotations after cropping images
    
    """
    jsonlist=os.listdir(annopath)
    for jsonname in jsonlist:
        if 'crop' in jsonname or 'VIVA' not in jsonname or 'json' not in jsonname:
            continue
        with open(os.path.join(annopath,jsonname),'r') as fp1:
            annodict=json.load(fp1)
        newdict=crop_single_anno(annodict,ori_size,resized,rejectsize)
        with open(os.path.join(annopath,jsonname.replace('.json','_crop.json')),'w') as fp2:
            json.dump(newdict,fp2,sort_keys=True, indent=4)
        print('done cropping annotation: '+jsonname)
    return newdict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k', 
        help="select the file path for BDD image and annotations")
    parser.add_argument('--crop_image', type=bool, 
        default=False, help="crop image or not, default as true")
    parser.add_argument('--crop_anno', type=bool, 
        default=True, help="crop annotations or not, default as true")
    parser.add_argument('--reject_size', type=int, 
        default=22, help="filter the bbox smaller than this size")
    # debug path:
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k
    # actual path:
    # D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    args = parser.parse_args()
    filepath=args.file_path
    cropimgflag=args.crop_image
    cropannoflag=args.crop_anno
    rejectsize=args.reject_size
    imagepath=os.path.join(filepath,'images')
    annopath=os.path.join(filepath,'labels')
    
    if cropimgflag:
        crop_images(imagepath)
    if cropannoflag:
        newdict=crop_annos(annopath=annopath,rejectsize=rejectsize)
    





""" End of file """
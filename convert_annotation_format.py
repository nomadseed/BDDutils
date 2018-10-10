# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:42:03 2018

convert the BDD (berkley deep drive dataset) annotation into VIVA annotator format

@author: Wen Wen
"""

import os
import json
import argparse

def convert_annotation_VIVA(annodict, preservedlabels={'car','motor','bus','truck'}):
    newdict={}
    for anno in annodict:
        newanno={}
        newanno['name']=anno['name']
        newanno['width']=1280
        newanno['height']=720
        newanno['annotations']=[]
        for bbx in anno['labels']:
            if bbx['category'] not in preservedlabels:
                continue
            else:
                newbox={}
                newbox['label']='car'
                newbox['id']=bbx['id']
                newbox['shape']=["Box", 1]
                newbox['category']='sideways'
                newbox['x']=int(bbx['box2d']['x1'])
                newbox['y']=int(bbx['box2d']['y1'])
                newbox['width']=int(bbx['box2d']['x2'])-int(bbx['box2d']['x1'])
                newbox['height']=int(bbx['box2d']['y2'])-int(bbx['box2d']['y1'])
                
                # append current bbx to the image anno
                newanno['annotations'].append(newbox)
        
        
        
        # push the anno of a single image into the dict
        newdict[anno['name']]=newanno
        
    return newdict
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k', 
        help="select the file path for image folders")
    args = parser.parse_args()
    filepath=args.file_path
    
    labelpath=os.path.join(filepath,'labels')
    jsonlist=os.listdir(labelpath)
    
    # load annotation files one by one and convert them into VIVA annotator format
    for jsonname in jsonlist:
        if '.json' not in jsonname:
            continue
        with open(os.path.join(labelpath,jsonname),'r') as fp:
            annodict=json.load(fp)
        
        newdict=convert_annotation_VIVA(annodict)
        newname=jsonname.replace('.json','_VIVA_format.json')
        
        with open(os.path.join(labelpath,newname),'w') as fp:
            json.dump(newdict,fp,sort_keys=True, indent=4)

        print(newname+' saved under '+labelpath)




            
""" End of File """
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
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

savedKmeanList={'ssd_opt_gt22':[
                    [16,13],[23,15],[18,20],
                    [25,20],[50,23],[33,38],
                    [51,47],[68,68],[43,65],[110,34],
                    [115,107],[153,63],[99,78],[78,101],
                    [146,126],[200,161],[216,96],[128,170],
                    [221,230],[180,280],[291,174],[271,260],[251,195]                    
                    ],
                'ssd_clust23p':[
                    [6,6],
                    [18,13],[11,10],
                    [33,27],[29,20],[17,22],
                    [33,51],[71,20],[41,35],
                    [52,52],[55,77],[114,22],[71,60],
                    [84,84],[159,57],[120,134],[165,113],[109,63],[236,77],
                            [120,97],[231,161],[153,178],[83,112]
                    ],
                'ssd_clust23p_gt22':[
                    [17,15],
                    [26,20],[36,30],[22,29],[30,44],
                    [65,59],[44,61],[57,83],[48,43],[79,26],
                    [80,81],[130,34],[100,61],[105,93],[139,80],[83,115],
                    [180,65],[250,80],[125,160],[171,203],[175,131],[128,119],
                    [246,155]
                    ],
                # the original anchors are acquired from 'multiple_grid_anchor_generator.py'
                # basesize=256
                # num_layers=6,
                # min_scale=0.2,
                # max_scale=0.95
                # aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3)
                'ssd_origin':[
                    [51, 51], [89, 89], [128, 128], [166, 166], [204, 204], 
                    [243, 243], [72, 36], [126, 63], [180, 90], [235, 117], 
                    [289, 144], [343, 171], [88, 29], [155, 51], [221, 73], 
                    [288, 96], [354, 118], [421, 140], [36, 72], [63, 126], 
                    [90, 180], [117, 235], [144, 289], [171, 343], [29, 88], 
                    [51, 155], [73, 221], [96, 288], [118, 354], [140, 421]
                    ],
                'ssd_adjusted':[
                    [24,8], [45,14], [90,29], [ 150,48], [ 225,72], 
                    [ 450,144], [ 15,12], [ 29,23], [ 57,46], [ 95,76], 
                    [ 143,114], [ 285,228], [ 12,15], [ 23,29], [ 45,58], 
                    [ 75,96], [ 113,144], [ 225,288], [ 10,18], [ 19,34], 
                    [ 38,68], [ 64,113], [ 96,169], [ 192,338], [ 9,20], 
                    [ 17,38], [ 34,76], [ 57,127], [ 85,191], [ 170,382]
                    ]
                }


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

def returnBoxEdge(box):
    # return area of box
    return box[0]+box[1]

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
                        bboxlist.append([round(anno['width']*0.46875),round(anno['height']*0.625)])
                
    return bboxlist

def groupBBox(bboxlist,threshlist,groupshape):
    """
    group bbox into different groups by grid size, save all the bbox into a dict
    that contain multiple bbox lists
    
    """
    xlist=bboxlist.copy()
    bboxdict={i:[] for i in range(len(threshlist))}
    print('grouping bbox based on threshlist, method: '+groupshape)
    if groupshape not in ['sector','belt','sickle']:
        raise ValueError('groupshape can only be belt or sector')
        
    xlist.sort(key=returnBoxArea,reverse=True)
    for i in range(len(xlist)):
        box=xlist.pop()    
        for layer in range(len(threshlist)):
            if groupshape=='tick':
                if box[0]*box[1]>=threshlist[layer][0]**2 and box[0]*box[1]<threshlist[layer][1]**2:# layer 0-5
                    bboxdict[layer].append(box)
            elif groupshape=='sector':
                if box[0]**2+box[1]**2>=2*threshlist[layer][0]**2 and box[0]**2+box[1]**2<2*threshlist[layer][1]**2:# layer 0-5
                    bboxdict[layer].append(box)
            elif groupshape=='belt':
                if box[0]+box[1]>=threshlist[layer][0]*2 and box[0]+box[1]<threshlist[layer][1]*2:# layer 0-5
                    bboxdict[layer].append(box)
                        

    
    return bboxdict

def simpleKMeans(xlist,n_cluster=2,random_state=0,algorithm='auto'):
    """
    
    output:
        centroids: centroids acquired from clustering
        labels: clustering result of input array
        score: sum of distances from centroids to all the points in their cluster
        time: processing time of current k-means clustering
        algorithm:
            'full' for classical EM-style algorithm
            'elkan' is more efficient by using the traiangle inequality, but 
                not suitable for sparse data.
            'auto' will choose elkan for dense data and full for sparse data
        
    """
    if len(xlist)==0:#skip empty list
        return [],[],1,0
    
    starttime=time.time()
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state,algorithm=algorithm).fit(xlist)
    processtime = time.time()-starttime
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = -kmeans.score(xlist)
    
    return centroids, labels, score, processtime

def adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.8,
                   threshtype='unified'):
    """
    take in a bboxlist and perform k-means clustering where k ranges from 2 to
    a maximum (defalut is 10). 
    Using Elbow method to get the optimized k value for the bboxlist, where the
    elbow point is defined by a score threshold. when current score is larger
    than last score, i.e. score>scorethresh*last_score, the last point is the
    elbow point. note that the changing threshold will have huge impact to the 
    final result.
    
    in the clusterlist, all the default boxes are saved as [width, height]
    
    """
    clusterlist=[]
    clusterlist_2=[]
    sumscore=0
    groupdict={}
    for l in bboxdict:
        grouplist=[]
        lastscore=float('Inf')
        notsaved=True
        for n_cluster in range(2,max_num_center):
            # k-means clustering        
            centroids, labels, score, processtime = simpleKMeans(xlist=bboxdict[l], n_cluster=n_cluster)
            print('clustering for k={}, processing time={}, score={}'.format(n_cluster,processtime,score))
            singledict={}
            singledict['k']=n_cluster
            if len(centroids)==0:
                singledict['centroids']=[]
                singledict['labels']=[]
            else:
                singledict['centroids']=centroids.tolist()
                singledict['labels']=labels.tolist()
            singledict['score']=score
            sumscore+=score
            singledict['processtime']=processtime
            grouplist.append(singledict)
            print('current score is {}% of last score'.format(score/lastscore))
            if notsaved and score/lastscore>=scorethresh:
                clusterlist.append({'thresh':l, 'final_centro':singledict['centroids'],'totalscore':sumscore})        
                notsaved=False
            lastscore=score
        groupdict[l]=grouplist

    # return k values with two methods of applying threshed
    
    # first method is apply unified value for all the layers
    if threshtype=='no_thresh':
        # this branch is for using threshtype 'all', not 'mean' nor 'abs'
        # which does not need effcost
        return clusterlist, groupdict, []
    
    elif threshtype=='unified':
        print('unified thresholds are applied')
        scorelist=[i['totalscore'] for i in clusterlist]
        layer=np.array([19*19,10*10,5*5,3*3,2*2,1*1])
        centro=np.array([len(i['final_centro']) for i in clusterlist])
        priors=np.sum(layer*centro)
        centronum=sum(centro)
        score=sum(scorelist)
        totalentities=sum([len(bboxdict[i]) for i in bboxdict])
        #avgscore=score/centronum
        #print('avgscore: {}'.format(avgscore))
        effcost=(score*priors)/(centronum*totalentities)
        print('KMeans thresh={}'.format(scorethresh))
        print('centro number: {}'.format(centronum))
        print('priors: {}'.format(priors))
        print('prior efficiency cost: {}'.format(effcost))
        
        return clusterlist, groupdict, []
    
    # second method is to find the combination of k values for the layers that
    # creates the lowest effcost
    elif threshtype=='argmin_effcost':
        """
        process the groupdict and minimize the effcose, return the k values
        for each layer, k ranges from [2,6], hence there are 5^6=15625 
        combinations to check. to reduce 
        
        """
        print('argmin thresholds are applied')
        layer=[19*19,10*10,5*5,3*3,2*2,1*1]
        
        sortresultlist=[]
        for l in range(0,6):
            #find the combination for the current layer that creates minimum effcost    
            totalentities=len(bboxdict[l])
            kmlist=groupdict[l]
            centros=np.array([i['k'] for i in kmlist]) # all k values of current layer
            scores=np.array([i['score'] for i in kmlist])
            priors=layer[l]*centros
            
            
            """
            some works need to be done on this single formula
            """
            effcost=scores*priors
            
            # create keyword array to sort with effcost
            dtype=[('k',int),('effcost',float)]
            values=[(i,j) for i,j in zip(centros,effcost)]
            kwdarr=np.array(values,dtype=dtype)
            sortresult=np.sort(kwdarr,order='effcost')
            sortresultlist.append(sortresult)
            
            opt_k, opt_score = sortresult[0]
            print('for {}th layer, k={}, score={}'.format(l,opt_k,opt_score))
            
            #save the results to clusterlist No.2
            for i in range(0,len(groupdict[l])):
                if groupdict[l][i]['k']==opt_k:
                    clusterlist_2.append({'thresh':l,
                                'final_centro':groupdict[l][i]['centroids'],
                                'totalscore':opt_score})
                    break
            
        return clusterlist_2, groupdict, sortresultlist
        
    
    
def plotScatter(bboxdict, clusterdict, savepath='',samecolor=False):
    chartaxis = [0.0,310.0,0.0,310.0]
    plt.figure(figsize=(7,6),dpi=100)
    plt.axis(chartaxis)
    if samecolor:
        colormap=['#0000ff','#0000ff']
    else:
        colormap=['#0000ff', '#00ff00']# use html colors
    centrolist=[i['final_centro'] for i in clusterdict]

    kwargs = {'family':'Times New Roman','fontsize':24}
    ax = plt.subplot(axisbg='white')
    
    # plot scatter for benchmarks
    for i, thresh in zip(range(len(bboxdict)),bboxdict):
        x_box = [i[0] for i in bboxdict[thresh]]
        y_box = [i[1] for i in bboxdict[thresh]]
        area_box = np.ones(len(x_box))*1
        ax.scatter(x_box, y_box, s=area_box, alpha=0.3, c=colormap[i%2], marker='o')# colormap[i]*len(x_box)
    

    # plot centroids
    for anchorlist in centrolist:
        x_anchor = [i[0] for i in anchorlist]
        y_anchor = [i[1] for i in anchorlist]
        area_anchor = np.ones(len(anchorlist))*100
        ax.scatter(x_anchor, y_anchor, s=area_anchor, alpha=1, c='r', marker='.')
    
    ax.set_xlabel('Width',**kwargs)
    ax.set_ylabel('Height',**kwargs)
    plt.title('Distribution of Ground Truth & Default Boxes',**kwargs)
    #plt.grid(True)
    plt.savefig(os.path.join(savepath,'scatter.png'),dpi=100)
    print('scatter saved as\n {}'.format(os.path.join(savepath,'scatter.png')))
    #plt.show()
    #plt.close()

def plotScatterSaved(bboxdict,clusterlist_1,clusterlist_2,
                     savepath):
    """
    this is a duplication of the function 'plotScatter', but take the saved
    list of kmeans centroids instead, this is a function for pure comparison
    
    """
    
    chartaxis = [0.0,310.0,0.0,310.0]
    plt.figure(figsize=(7,6),dpi=100)
    plt.axis(chartaxis)
    kwargs = {'family':'Times New Roman','fontsize':24}
    
    ax = plt.subplot(axisbg='white')
    
    # plot scatter for benchmarks
    for i, thresh in zip(range(len(bboxdict)),bboxdict):
        x_box = [i[0] for i in bboxdict[thresh]]
        y_box = [i[1] for i in bboxdict[thresh]]
        area_box = np.ones(len(x_box))*1
        ax.scatter(x_box, y_box, s=area_box, alpha=0.1, c='b', marker='o')
        
    # plot centroids of cluster result 1 in red
    for anchor in savedKmeanList[clusterlist_1]:
        x_anchor=anchor[0]
        y_anchor=anchor[1]
        ax.scatter(x_anchor, y_anchor, s=30, alpha=1, c='r', marker='o')

    # plot centroids of cluster result 2 in green
    if clusterlist_2 is not None:
        for anchor in savedKmeanList[clusterlist_2]:
            x_anchor=anchor[0]
            y_anchor=anchor[1]
            ax.scatter(x_anchor, y_anchor, s=30, alpha=1, c='g', marker='^')

    ax.set_xlabel('Width',**kwargs)
    ax.set_ylabel('Height',**kwargs)
    plt.title('Raw & Layer-wise K-means',**kwargs)
    #plt.grid(True)
    plt.savefig(os.path.join(savepath,'kmeans_comparison.png'),dpi=100)
    print('scatter saved as\n {}'.format(os.path.join(savepath,'kmeans_comparison.png')))
    if clusterlist_2 is not None:
        print('red dots are {}, green triangles are {}'.format(clusterlist_1,
                                         clusterlist_2))
        
                    
    #plt.show()
    #plt.close()

PATH_DICT={'bdd':{'path':'D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k',
                  'label':'bdd100k_labels_images_train_VIVA_format_crop_gt22.json'},
           'cal':{'path':'D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset',
                  'label':'caltech_annotation_VIVA_train.json'}
           }

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default=PATH_DICT['bdd']['path'], 
                        help="File path of input data")
    #D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    #D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset
    
    parser.add_argument('--json_label', type=str, 
                        default=PATH_DICT['bdd']['label'], 
                        help="label to specify json file")
    #bdd100k_labels_images_val_VIVA_format_crop.json
    #caltech_annotation_VIVA_test.json
    
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    savepath = os.path.join(filepath,'ssd_cluster_result')
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get bboxlist from annotation
    bboxlist = get_bboxes(annotationdict)
    
    # setup threshlist, where threshold is diagnal length in pixel
    # to get the threshlist easily, check the get_anchor_from_bench_mark.py
    # run the file and check h_pel_list
    
    # SETUPS
    #threshtype='abs'
    #threshtype='mean'
    threshtype='all' # if use this, please change 'threshtype' in adaptiveKMeans to 'no_thresh'
    scorethresh=0.7 # default=0.7
    #groupshape='belt'
    groupshape='sector'
    #groupshape='sickle'
    compareonly=True
    
    
    '''threshes for various dataset'''
    if threshtype=='abs':
        # Threshlist for BDD dataset, of grid size [19,10,5,3,2,1]
        # Matching area size: [16,30,60,100,150,300]
        threshlist = [[0,22],
                      [22,42],
                      [42,85],
                      [70,141],
                      [106,212],
                      [212,10000]
                      ] 
    elif threshtype=='mean':
        # Threshlist for BDD dataset, of grid size [19,10,5,3,2,1], 
        # but use mean of grid sizes as boundary such that there is no overlapping
        # between two adjacent box groups
        threshlist = [[0,22],
                      [22,42],
                      [42,(85+70)/2],
                      [(85+70)/2,(141+106)/2],
                      [(141+106)/2,212],
                      [212,10000]
                      ] 
    #threshlist=[13.63, 17.64, 33.33, 75, 150, 500] # Threshlist for Caltech dataset, of grid size [22,17,9,4]
    elif threshtype=='all':
        threshlist=[[0,10000]]
        scorethresh=0.7
    
    bboxdict = groupBBox(bboxlist,threshlist,groupshape=groupshape)
    
    # perform k-means for all the layers
    if not compareonly:
        clusterdict, groupdict, sortresultlist = adaptiveKMeans(bboxdict,
                                                                max_num_center=10,
                                                                scorethresh=scorethresh,
                                                                threshtype='no_thresh')#max_num_center=25
        json.dump(clusterdict, open(os.path.join(savepath,'k-means_cluster_result_ssd.json'), 'w'),sort_keys=True, indent=4)

    # draw figures of [sum of error over k values]
    if not compareonly:
        plotScatter(bboxdict, clusterdict, savepath=savepath,samecolor=True)
        print('Layer-wise KMeans for {}-{} method'.format(groupshape,threshtype))
        
        
    else:
        plotScatterSaved(bboxdict,
                         clusterlist_1='ssd_clust23p_gt22',
                         clusterlist_2=None,
                         savepath=savepath)

    


"""End of file"""
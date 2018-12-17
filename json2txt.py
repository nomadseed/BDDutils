# coding = utf-8
import json
import numpy as np
import os
import argparse

classes = ['car']

def json2txt(mainfolder,
             train_txt="train_5.txt",
             val_txt='val_5.txt',
            json_name = None,
            train_factor=0.8):

    train_f = open(mainfolder+train_txt, 'w')
    val_f = open(mainfolder + val_txt,'w')


    json_name = mainfolder+'train_boxes.json'
    train_number=0
    val_number=0
    #print(json_name)
    with open(json_name) as load_f:
        load_dict = json.load(load_f)
        np.random.shuffle(load_dict)
        for i in range(len(load_dict)):
            pic = load_dict[i]
            if i%5==0:
                if i < train_factor*len(load_dict):# train_factor*len(load_dict) is the size of the trainset
                    train_f.write(mainfolder+pic['image_path']+' ')
                    for rect in pic['rects']:
                        x1,y1,x2,y2=int(rect['x1']),int(rect['y1']),int(rect['x2']),int(rect['y2'])
                        train_f.write(str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+',0 ')
                    train_f.write('\n')
                    train_number+=1

                else:
                    val_f.write(mainFolder + pic['image_path'] + ' ')
                    for rect in pic['rects']:
                        val_f.write(
                            str(int(rect['x1'])) + ',' + str(int(rect['y1'])) + ',' + str(int(rect['x2'])) + ',' + str(
                                int(rect['y2'])) + ',0 ')
                    val_f.write('\n')
                    val_number+=1
    train_f.close()
    val_f.close()
    print("train",train_number,"val",val_number)

def car_json2txt(
             savepath='',
             imgpath='',
             jsonname = None,
             outputname = None
                 ):
    '''
    param savepath: path to save txt file
    param imgpath: path of images, for saving text in txt file as keys 
    param jsonname: json file to be loaded
    param outputname: specified name for output file, if not specified, use txt
        to substitute the json
    '''
    
    # the name of the benchmark
    if jsonname==None:
        assert ValueError('Name of jsonfile is not provided')
            
    if outputname is None:
        outputname=jsonname.replace('.json','.txt')
    txt_f = open(os.path.join(savepath,outputname), 'w')

    imgcount = 0

    # go through all the annotation file
    namelist=os.listdir(savepath)
    
    # open the annotation file and shuffle
    with open(jsonname) as load_f:
        load_dict = json.load(load_f)
        images = list(load_dict.keys())
        np.random.shuffle(images)
        for imgname in images:
            rects = load_dict[imgname]['annotations']
            # train_factor*len(load_dict) is the size of the trainset
            txt_f.write(str(os.path.join(imgpath,imgname).replace('\\','/'))+' ')
            for rect in rects:
                x1,y1,x2,y2=int(rect['x']),int(rect['y']),int(rect['x']+rect['width']),int(rect['y']+rect['height'])
                txt_f.write(str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+',0 ')
            txt_f.write('\n')
            imgcount+=1

    txt_f.close()
    print('imgpath: {} \n savepath: {}'.format(imgpath,savepath))
    print("{} images processed".format(imgcount))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path',type=str,default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/labels/bdd100k_labels_images_val_VIVA_format_crop.json',
                        help='json file to be converted, default is None')
    parser.add_argument('--img_path',type=str,default='/home/wenwen/BDD_car/bdd100k/images/100k_cropped/train',
                        help='path of corresponding images')
    parser.add_argument('--save_path',type=str,default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k/labels',
                        help='path of annotation file')
    arg=parser.parse_args()
    
    # mainFolder = '../caltech_ped/caltech-pedestrian-dataset-converter/'
    # json2txt(mainFolder)
    savepath = arg.save_path
    jsonname=arg.json_path
    imgpath=arg.img_path
    
    car_json2txt(savepath=savepath,jsonname=jsonname,imgpath=imgpath)

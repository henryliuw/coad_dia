import openslide
import os
# if line 6 encounter error run following cmd in bash:
#export PYTHONPATH=/home/DiskA/liuhongyi/coad_dia/kfbreader:$PYTHONPATH
#export LD_LIBRARY_PATH=~/anaconda3/lib:/home/DiskA/liuhongyi/coad_dia/kfbreader:$LD_LIBRARY_PATH
from kfbreader import kfbReader
import numpy as np
import matplotlib.image
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
import pickle
import gc
import time
import pandas as pd
import random
from extractor import MyResNet
import sys
sys.path.append('Vahadane-master')
import spams
import utils
from vahadane import vahadane
from sklearn.manifold import TSNE
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings('ignore')  # possibly harmful code

def Otsu_threshold(img, verbose=False):
    ''' 
    to detect/segment useful part in WSI
    return a boolean mask using Otsu thresholding in HSV space given an image in RGB space
    if verbose plot the mask
    '''
    # transfer to HSV color space
    img_bgr = img[:,:,::-1]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ret_h, th_h = cv2.threshold(img_hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret_s, th_s = cv2.threshold(img_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (th_h * th_s != 0)
    if verbose:
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(20, 10))
        ax0.imshow(img)
        ax0.set_title("Original image")
        ax1.imshow(th, cmap='gray')
        ax1.set_title("Otsu thresholding")
        fig.tight_layout()
    return th

def SPCN(source, verbose=False):
    if not hasattr(SPCN, 'Wt') or not hasattr(SPCN, 'Ht'):
        print("Initializing stain normalization")
        target = utils.read_image('Vahadane-master/output/reference.jpg')
        vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
        vhd.fast_mode=0
        vhd.getH_mode=0
        SPCN.Wt, SPCN.Ht = vhd.stain_separate(target, verbose)
        SPCN.target = target
    else:
        vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
        vhd.fast_mode=0
        vhd.getH_mode=0
    Ws, Hs = vhd.stain_separate(source, verbose)
    img = vhd.SPCN(source, Ws, Hs, SPCN.Wt, SPCN.Ht)
    if verbose:
        plt.figure(figsize=(30, 10))
        plt.subplot(1,3,1)
        plt.title('Source', fontsize=50)
        plt.imshow(source)
        plt.subplot(1,3,2)
        plt.title('Target', fontsize=50)
        plt.imshow(SPCN.target)
        plt.subplot(1,3,3)
        plt.title('Result', fontsize=50)
        plt.imshow(img)
        plt.show()
    return img

def features_extraction(img, feature_extractor):
    '''
    given an image (ndarray or PIL image), return a [1, 2048] dimensional ndarray feature vector 
    '''
    if type(img) is np.ndarray:
        img=Image.fromarray(img)

    transform_train = transforms.Compose([ 
        transforms.RandomCrop(224), 
        transforms.ToTensor(), 
        #lambda x: x / 255,
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
        transforms.Normalize(mean=(0.71029204, 0.54590117, 0.64897074), std=(0.19687233, 0.22803243, 0.1713212))
    ])
    img_transformed = transform_train(img).reshape(1,3,224,224)
    features = feature_extractor(img_transformed).view(1, -1).detach().numpy()
    return features

def data_augmentation_transform(img):
    vertical_flip = random.choice([0,1])
    horizontal_flip= random.choice([0, 1])
    rotate_seed = random.choice([1,2,3,4])

    if vertical_flip:
        img = np.flipud(img)
    if horizontal_flip:
        img = np.fliplr(img)
    img = np.rot90(img, rotate_seed)    
    return img

def read_samples(image_path, save_dir, name, sample_size, threshold_ratio=0.3, evaluate=True, source='tcga', extractor=""):
    if '.svs' in image_path:
        slide = openslide.OpenSlide(image_path)
    
        level_downsamples = sorted([round(i) for i in slide.level_downsamples], reverse=True)
        if 32 not in level_downsamples and 64 not in level_downsamples:
            print('File %s does not have downsample levels larger than or equal to 32' % image_path)
            return None
        low_resolution_img = np.array(slide.read_region((0,0), len(slide.level_dimensions)-1, slide.level_dimensions[-1]))
    elif '.kfb' in image_path:
        read0 = kfbReader.reader()
        read0.ReadInfo(image_path, 1, False)
        low_resolution_img = read0.ReadRoi(0,0, read0.getWidth() , read0.getHeight() , scale=1)
        read = kfbReader.reader()
        read.ReadInfo(image_path, 32, False)
        
    # print(level_downsamples, slide.level_dimensions)
    # print(slide.level_dimensions[0][0] * slide.level_dimensions[0][1])
    # print(slide.level_dimensions[-1][0] * slide.level_dimensions[-1][1])
    # return 
    
    width, height = low_resolution_img.shape[0]//7, low_resolution_img.shape[1]//7
    mask_ij = np.zeros((width, height), np.bool_)
    Otsu_mask = Otsu_threshold(low_resolution_img[:,:,:3], False)
    tile_list = []

    for i in range(width):
        for j in range(height):
            ratio = Otsu_mask[i*7:(i+1)*7, j*7:(j+1)*7].sum()/7/7
            # print(ratio)
            if ratio > threshold_ratio:
                mask_ij[i][j] = True
                tile_list.append((i, j))
    #if evaluate:
    #    sample_size = len(tile_list)
    print(len(tile_list))
    #return
    
    random.shuffle(tile_list)
    resnet34 = MyResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3])
    resnet34.load(extractor)
    resnet34.eval()
    feature_extractor = resnet34.get_feature
    
    #resnet50_model = torchvision.models.resnet50(pretrained=True)
    #resnet50_model.eval()
    #feature_extractor = torch.nn.Sequential(*list(resnet50_model.children())[:-1])
    #model = ResNet18()
    #model.load_state_dict(torch.load('data/Resnet34_fcn_best.pkl'))
    #model = model.eval()
    #feature_extractor = model.forward_feature

    feature_vec = None
    name_list = []
    idx = 0
    count = 0
    if not os.path.isdir(save_dir): # short
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, source)
    name_i = name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if evaluate:
        if not os.path.isdir(os.path.join(save_dir, name_i)):
            os.mkdir(os.path.join(save_dir, name_i))
        if not os.path.isdir(os.path.join(save_dir, name_i,'discarded')):
            os.mkdir(os.path.join(save_dir, name_i,'discarded'))

    print('sampling %d tiles from image %s' % (sample_size, name))
    while 1:
        # get idx
        try:
            j, i = tile_list[idx]
        except IndexError: # unsuccessful exit branch
            print('image %s does not have enough tiles for sampling' % name)
            break

        pic_name = '%d-%d' % (i, j)
        # read
        if '.svs' in image_path:
            if 32 in level_downsamples:
                idx_32 = level_downsamples.index(32)
                img = np.array(slide.read_region((224 * i * 2 ** idx_32, 224 * j * 2**idx_32), idx_32, (224, 224)))
            elif 64 in level_downsamples:
                idx_64 = level_downsamples.index(64)
                img_PIL = slide.read_region((224 * 2 * i * 2 ** idx_64, 224 * 2 * j * 2**idx_64), idx_64, (224 * 2, 224 * 2))
                img =  np.asarray(img_PIL.resize((224, 224))) # resize
        elif '.kfb' in image_path:
            img = read.ReadRoi(224 * i, 224 * j, 224, 224, scale=32)
        # feature extraction
        try:
            img = img[:, :, :3]
            img_SN = SPCN(img.copy())
            #img_BN, H, E = normalizeStaining(img)
            img_SN = data_augmentation_transform(img_SN)
            if evaluate:
                matplotlib.image.imsave(save_dir+'/'+name_i+'/'+pic_name+'.png', img_SN)
            features = features_extraction(img_SN, feature_extractor)
            name_list.append((i,j))
            if feature_vec is None:
                feature_vec = features
            else:
                feature_vec = np.r_[feature_vec, features]
            idx+=1
            count += 1
        except:
            # 
            if evaluate:
                print('item %s is discarded for %s' % (pic_name, name))
                matplotlib.image.imsave(save_dir+'/'+name_i+'/discarded/'+pic_name+'.png', img)
            idx+=1
            continue
                # check for stop condition first
        # successful exit branch
        if count == sample_size or idx == len(tile_list):
            # save files
            with open(os.path.join(save_dir,name_i+'_name.pkl'),'wb') as file:
                pickle.dump(name_list, file)
            np.save(os.path.join(save_dir, name_i + '_features.npy'), feature_vec)
            feature_vec = None
            name_list = []
            count = 0
            #if not os.path.isdir(os.path.join(save_dir, name_i)):
            #    os.mkdir(os.path.join(save_dir, name_i))
            #if not os.path.isdir(os.path.join(save_dir, name_i,'discarded')):
            #    os.mkdir(os.path.join(save_dir, name_i, 'discarded'))
            print('sampling preprocessing succeed')
            break
    if evaluate:
        return mask_ij, low_resolution_img, feature_vec, name_list
    else:
        slide.close()
        del name_list
        del feature_vec


def test():
    image_path = 'data/TCGA-0.svs'
    save_dir = 'data'
    name ='sample-test'
    read_samples(image_path, save_dir, name, 50, repl_n=3)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='data/test' ,help='determine the output dir of preprocessed data')
    parser.add_argument("--sample_n", default=2000, type=int, help='sample size of each image')
    parser.add_argument("--source", default='TCGA', help='one of TCGA/changhai/TH')
    parser.add_argument("--extractor", default="f32", help='name to load extractor')
    args = parser.parse_args()
    # preprocessing
    if args.source=='TCGA':
        useful_subset = pd.read_csv('data/useful_subset.csv')
        for i in useful_subset.index:
            image_path = os.path.join('/home/DiskB/tcga_coad_dia', useful_subset.loc[i, 'id'], useful_subset.loc[i, 'File me'])
            name = str(i)
            print('%s\tstarting image %d' % (time.strftime('%Y.%m.%d.%H:%M:%S',time.localtime(time.time())), i))
            read_samples(image_path,  args.output_dir, name, sample_size=args.sample_n, source='tcga', extractor=args.extractor)
    elif args.source=='changhai':
        data_xls = pd.ExcelFile('data/information.xlsx')
        df = data_xls.parse(sheet_name='changhai')
        for i in df.index:
            if df.loc[i, 'use']:
                image_path = os.path.join('/home/DiskB/COAD_additional_data/changhai', str(df.loc[i, 'filename'])+'.svs')
                read_samples(image_path,  args.output_dir, str(i), sample_size=args.sample_n, source='changhai', extractor=args.extractor)
    elif args.source=='TH':
        data_xls = pd.ExcelFile('data/information.xlsx')
        df = data_xls.parse(sheet_name='TumorHospital')
        for i in df.index:
            if df.loc[i, 'use']:
                file_id = df.loc[i, 'filename'][2:]
                image_files = os.listdir('/home/DiskB/COAD_additional_data/TumorHospital')
                for image_file in image_files:
                    if file_id in image_file and '.kfb' in image_file:
                        image_path = os.path.join('/home/DiskB/COAD_additional_data/TumorHospital', image_file)
                        read_samples(image_path, args.output_dir, file_id, sample_size=args.sample_n, source='TH', extractor=args.extractor)
        
if __name__ == "__main__":
    main()
    #test()

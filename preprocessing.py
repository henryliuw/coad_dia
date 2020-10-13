import openslide
import numpy as np
import matplotlib.image
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision
import torch
import pickle
import gc
import time
import pandas as pd
import random
from extractor import MyResNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors of the covariance between each channel of RGB
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues, which is sorted by ascending value
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile+'.png')
        Image.fromarray(H).save(saveFile+'_H.png')
        Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E

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

def read_image(slide, slide_count):
    level_downsamples = sorted([round(i) for i in slide.level_downsamples], reverse=True)
    if 32 in level_downsamples:
        idx_32 = level_downsamples.index(32)
        for i in range(slide.level_dimensions[idx_32][0]//(224 * slide_count)):
            img = np.array(slide.read_region((224 * slide_count * i * 2**idx_32,0), idx_32, (224 * slide_count, slide.level_dimensions[idx_32][1])))
            yield img, i
    elif 64 in level_downsamples:
        idx_64 = level_downsamples.index(64)
        for i in range(6, slide.level_dimensions[idx_64][0]//(224 * slide_count)):
            img_PIL = slide.read_region((224 * 2 * slide_count * i * 2**idx_64,0), idx_64, (224 * 2 * slide_count, slide.level_dimensions[idx_64][1]))
            img_resized =  np.asarray(img_PIL.resize((img_PIL.size[0]//2, img_PIL.size[1]//2)))
            del img_PIL
            gc.collect()
            yield img_resized, i

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

def read_samples(image_path, save_dir, name, sample_size, repl_n=1, threshold_ratio=0.3, evaluate=False, changhai=False, extractor=""):
    slide = openslide.OpenSlide(image_path)
    
    level_downsamples = sorted([round(i) for i in slide.level_downsamples], reverse=True)
    if 32 not in level_downsamples and 64 not in level_downsamples:
        print('File %s does not have downsample levels larger than or equal to 32' % image_path)
        return None

    low_resolution_img = np.array(slide.read_region((0,0), len(slide.level_dimensions)-1, slide.level_dimensions[-1]))
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
    if evaluate:
        sample_size = len(tile_list)
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
    repl_i = 0
    if changhai:
        save_dir = os.path.join(save_dir, 'changhai')
    name_i = name+'_'+str(repl_i)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    #if not os.path.isdir(os.path.join(save_dir, name_i)):
    #    os.mkdir(os.path.join(save_dir, name_i))
    #if not os.path.isdir(os.path.join(save_dir, name_i,'discarded')):
    #    os.mkdir(os.path.join(save_dir, name_i,'discarded'))

    print('sampling %d tiles from image %s' % (sample_size, name))
    while 1:
        # check for stop condition first
        # successful exit branch
        if count == sample_size or idx == len(tile_list):   
            with open(os.path.join(save_dir,name_i+'_name.pkl'),'wb') as file:
                pickle.dump(name_list, file)
            np.save(os.path.join(save_dir, name_i + '_features.npy'), feature_vec)
            feature_vec = None
            name_list = []
            count = 0
            repl_i += 1
            name_i = name+'_'+str(repl_i)
            #if not os.path.isdir(os.path.join(save_dir, name_i)):
            #    os.mkdir(os.path.join(save_dir, name_i))
            #if not os.path.isdir(os.path.join(save_dir, name_i,'discarded')):
            #    os.mkdir(os.path.join(save_dir, name_i, 'discarded'))
            print('replica %d sampling succeed' % repl_i)
            if repl_i == repl_n:
                break
        # get idx
        try:
            j, i = tile_list[idx]
        except IndexError: # unsuccessful exit branch
            print('image %s does not have enough tiles for sampling for replica %d' % (name, repl_i))
            break

        pic_name = '%d-%d' % (i, j)
        # read
        if 32 in level_downsamples:
            idx_32 = level_downsamples.index(32)
            img = np.array(slide.read_region((224 * i * 2 ** idx_32, 224 * j * 2**idx_32), idx_32, (224, 224)))
        elif 64 in level_downsamples:
            idx_64 = level_downsamples.index(64)
            img_PIL = slide.read_region((224 * 2 * i * 2 ** idx_64, 224 * 2 * j * 2**idx_64), idx_64, (224 * 2, 224 * 2))
            img =  np.asarray(img_PIL.resize((224, 224))) # resize
        # feature extraction
        try:
            img = img[:, :, :3]
            img_BN, H, E = normalizeStaining(img)
            img_BN = data_augmentation_transform(img_BN)
            if evaluate:
                matplotlib.image.imsave(save_dir+'/'+name_i+'/'+pic_name+'.png', img)
            features = features_extraction(img_BN, feature_extractor)
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
    parser.add_argument("--input_dir", default='/home/DiskB/tcga_coad_dia'  ,help='determine the base dir of the dataset document')
    parser.add_argument("--output_dir", default='preprocessed_data' ,help='determine the output dir of preprocessed data')
    parser.add_argument("--start_image", default=0 , type=int, help='starting image index of preprocessing (for continuing unexpected break)')
    parser.add_argument("--sample_n", default=1000, type=int, help='sample size of each image')
    parser.add_argument("--repl_n", default=0, type=int, help='replication of each image')
    parser.add_argument("--extractor", default="", help='name to load extractor')
    parser.add_argument("--changhai", action='store_true')
    args = parser.parse_args()
    useful_subset = pd.read_csv('data/useful_subset.csv')
    # preprocessing
    if not args.changhai:
        for i in useful_subset.index:
            if i < args.start_image:
                continue
            image_path = os.path.join(args.input_dir, useful_subset.loc[i, 'id'], useful_subset.loc[i, 'File me'])
            name = str(i)
            print('%s\tstarting image %d' % (time.strftime('%Y.%m.%d.%H:%M:%S',time.localtime(time.time())), i))
            read_samples(image_path,  args.output_dir, name, sample_size=args.sample_n, repl_n=args.repl_n, extractor=args.extractor)
    else:
        for i in sorted(os.listdir('/home/DiskB/tcga_coad_dia/changhai')):
            image_path = os.path.join('/home/DiskB/tcga_coad_dia/changhai', i)
            read_samples(image_path,  args.output_dir, i.split('.')[0], sample_size=args.sample_n, repl_n=args.repl_n, changhai=True, extractor=args.extractor)
        
if __name__ == "__main__":
    main()
    #test()

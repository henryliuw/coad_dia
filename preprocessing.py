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
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
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

def preprocessing(image_path, save_dir, name, threshold_ratio=0.3):
    # reading, thresholding 
    slide = openslide.OpenSlide(os.path.join(image_path))
    # extract features
    resnet50_model = torchvision.models.resnet50(pretrained=True)
    resnet50_model.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50_model.children())[:-1])
    feature_vec = None
    name_list = []
    slide_count = 5
    gen = read_image(slide, slide_count)
    
    #get Otsu mask on low resolution image
    Otsu_mask = Otsu_threshold(np.array(
        slide.read_region((0,0),len(slide.level_dimensions)-1, slide.level_dimensions[-1]))[:,:,:3], False)
    
    while 1:
        try:
            img, slide_idx = next(gen)
            row_n = img.shape[0] // 224
            col_n = img.shape[1] // 224
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            if not os.path.isdir(os.path.join(save_dir,'discarded')):
                os.mkdir(os.path.join(save_dir,'discarded'))
            gc.collect()
            # matplotlib.image.imsave(save_dir+'/slide_'+str(slide_idx)+'.png', img)
            

            for i in range(row_n):
                for j in range(col_n):
                    mask_i = 7 * i # relative coordinate
                    mask_j = 7 * (slide_count * slide_idx + j)
                    mask_ij = Otsu_mask[mask_i:mask_i+7, mask_j:mask_j+7]
                    ratio = mask_ij.sum() / 7 / 7
                    # print(ratio)
                    if ratio > threshold_ratio:
                        grid_ij = img[224*i:224*(i+1),224*j:224*(j+1),:3]
                        plt.imshow(grid_ij)
                        try:
                            pic_name = '%d-%d' % (i, j + slide_idx * 5)
                            img_BN, H, E = normalizeStaining(grid_ij)
                            img_BN = grid_ij
                            matplotlib.image.imsave(save_dir+'/'+name+pic_name+'.png', img_BN)
                            features = features_extraction(img_BN, feature_extractor)
                            name_list.append(pic_name)
                            if feature_vec is None:
                                feature_vec = features
                            else:
                                feature_vec = np.r_[feature_vec, features]

                        except:
                            print('item %s is discarded for %s' % (pic_name, name))
                            matplotlib.image.imsave(save_dir+'/'+name+'/discarded/'+pic_name+'.png', grid_ij)
                            pass

            gc.collect()
        except StopIteration:
            break
    

    slide.close()
    with open(os.path.join(save_dir,name+'_name.pkl'),'wb') as file:
        pickle.dump(name_list, file)
    np.savetxt(os.path.join(save_dir,name+'_features.txt'),feature_vec)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='/home/DiskB/tcga_coad_dia'  ,help='determine the base dir of the dataset document')
    parser.add_argument("--output_dir", default='preprocessed_data' ,help='determine the base dir of the dataset document')
    args = parser.parse_args()
    useful_subset = pd.read_csv('useful_subset.csv')
    # preprocessing
    for i in useful_subset.index:
        image_path = os.path.join(input_dir, useful_subset.loc[i, 'id'], useful_subset.loc[i, 'File me'])
        name = str(i)
        print('%s\tstarting image %d' % (time.strftime('%Y.%m.%d',time.localtime(time.time())), i))
        preprocessing(image_path, args.output_dir, name)

if __name__ == "__main__":
    main()
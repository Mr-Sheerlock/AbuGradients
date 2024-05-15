from commonfunctions import *

import cv2
import numpy as np
import joblib
from skimage.feature import hog
from Preprocessing.preprocess import preprocess, removeSkew
import skimage as ski
import torch
import pickle as pkl

model=torch.load('font_classifier_model_pytorch_HoGLCM88_Deploy.pth')
# pca_HoG=joblib.load('pca_model_resize500_HoG_LoG_pca99_16x8.sav')
# pca_GLCM=joblib.load('pca_model_resize500_glcm_LoG_pca99.sav')
with open ('pca_model_resize500_HoG_LoG_pca99_16x8.sav', 'rb') as f:
    pca_HoG=pkl.load(f)

with open ('pca_model_resize500_glcm_LoG_pca99.sav', 'rb') as f:
    pca_GLCM=pkl.load(f)

def extract_hog_features(image):
    # Check if the input image is grayscale
    # gray = None
    # if len(image.shape) == 3:
    #     gray = rgb2gray(image)
    
    # Compute HOG features
    features = hog(image, orientations=9, pixels_per_cell=(16, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

def get_glcm_features(img):
    # 8-Neighbourhood angles 
    comatrix = ski.feature.graycomatrix(img, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast=ski.feature.graycoprops(comatrix, 'contrast')
    dissimilarity=ski.feature.graycoprops(comatrix, 'dissimilarity')
    homogeneity=ski.feature.graycoprops(comatrix, 'homogeneity')
    energy=ski.feature.graycoprops(comatrix, 'energy')
    correlation=ski.feature.graycoprops(comatrix, 'correlation')
    Toreturn=np.concatenate([contrast.ravel(), dissimilarity.ravel(), homogeneity.ravel(), energy.ravel(), correlation.ravel()])
    return Toreturn

resizefactor=500
img=io.imread('fonts-dataset/IBM Plex Sans Arabic/0.jpeg')
# img=cv2.imdecode(img, cv2.IMREAD_COLOR)
img=preprocess(img)
rot=removeSkew(img,resizefactor)
HoGfeatures=extract_hog_features(rot).reshape(1,-1)
HoGfeatures=np.array(HoGfeatures)
print(HoGfeatures.shape)
HoGfeatures = pca_HoG.transform(HoGfeatures)
print(HoGfeatures.shape)
glcm_features=get_glcm_features(rot).reshape(1,-1)
glcm_features=np.array(glcm_features)
print(glcm_features.shape)
glcm_features = pca_GLCM.transform(glcm_features)
print(glcm_features.shape)
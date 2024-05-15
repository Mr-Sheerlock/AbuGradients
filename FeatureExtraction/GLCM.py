import skimage as ski
import numpy as np
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
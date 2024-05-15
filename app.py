from flask import Flask, request, jsonify
from commonfunctions import *

import cv2
import numpy as np
import joblib
from skimage.feature import hog
from Preprocessing.preprocess import preprocess, removeSkew
import skimage as ski

app = Flask(__name__)

model=joblib.load('font_classifier_model_pytorch_HoGLCM88_Deploy.pth')
pca_HoG=joblib.load('pca_model_resize_500_glcm_LoG_pca99.sav')
pca_GLCM=joblib.load('pca_model_resize_500_HoG_LoG_pca99.sav')


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
@app.route('/predict', methods=['POST'])
def predict():
    try: 
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            img=np.frombuffer(file.read(), np.uint8)
            img=cv2.imdecode(img, cv2.IMREAD_COLOR)
            img=preprocess(img)
            rot=removeSkew(img,resizefactor)
            HoGfeatures=extract_hog_features(rot).reshape(1,-1)
            HoGfeatures=np.array(HoGfeatures)
            print(HoGfeatures.shape)
            HoGfeatures = pca_HoG.fit(HoGfeatures)
            glcm_features=get_glcm_features(rot).reshape(1,-1)
            glcm_features=np.array(glcm_features)
            print(glcm_features.shape)
            glcm_features = pca_GLCM.fit(glcm_features)

            # print("features after PCA ",HoGfeatures.shape)
            #Normalize the features
            allfeatures=np.concatenate((HoGfeatures,glcm_features),axis=1)
            prediction = model.predict(allfeatures)
            return jsonify({'prediction': str(prediction)}) 
    except Exception as e:
        return jsonify({"error":str(e)})
    
    
if __name__ =='__main__':
    app.run(debug=True)
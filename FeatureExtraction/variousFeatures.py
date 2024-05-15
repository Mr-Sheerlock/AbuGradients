import numpy as np

#%matplotlib inline
def whiteBlackRatio(img):
    h = img.shape[0]
    w = img.shape[1]
    #initialized at 1 to avoid division by zero
    blackCount=1
    whiteCount=0
    blackCount+=np.sum(img==0)
    whiteCount+=np.sum(img==255)
    return whiteCount/blackCount

def horizontalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for y in range(0,h):
        prev=img[y,0]
        transitions=0
        for x in range (1,w):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)
            
    return maximum
def verticalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for x in range(0,w):
        prev=img[0,x]
        transitions=0
        for y in range (1,h):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)
            
    return maximum

def histogramAndCenterOfMass(img):
    h = img.shape[0]
    w = img.shape[1]
    histogram=[]
    sumX=0
    sumY=0
    num=0
    for x in range(0,w):
        localHist=0
        for y in range (0,h):
            if(img[y,x]==0):
                sumX+=x
                sumY+=y
                num+=1
                localHist+=1
        histogram.append(localHist)
      
    return sumX/num , sumY/num, histogram

def getFeatures(img):
    x,y= img.shape
    featuresList=[]
    # first feature: height/width ratio
    featuresList.append(y/x)
    #second feature is ratio between black and white count pixels
    featuresList.append(whiteBlackRatio(img))
    #third and fourth features are the number of vertical and horizontal transitions
    featuresList.append(horizontalTransitions(img))
    featuresList.append(verticalTransitions(img))

    #print (featuresList)
    #splitting the image into 4 images
    topLeft=img[0:y//2,0:x//2]
    topRight=img[0:y//2,x//2:x]
    bottomeLeft=img[y//2:y,0:x//2]
    bottomRight=img[y//2:y,x//2:x]

    #get white to black ratio in each quarter
    featuresList.append(whiteBlackRatio(topLeft))
    featuresList.append(whiteBlackRatio(topRight))
    featuresList.append(whiteBlackRatio(bottomeLeft))
    featuresList.append(whiteBlackRatio(bottomRight))

    #the next 6 features are:
    #• Black Pixels in Region 1/ Black Pixels in Region 2.
    #• Black Pixels in Region 3/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 3.
    #• Black Pixels in Region 2/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 4
    #• Black Pixels in Region 2/ Black Pixels in Region 3.
    topLeftCount=blackPixelsCount(topLeft)
    topRightCount=blackPixelsCount(topRight)
    bottomLeftCount=blackPixelsCount(bottomeLeft)
    bottomRightCount=blackPixelsCount(bottomRight)

    featuresList.append(topLeftCount/topRightCount)
    featuresList.append(bottomLeftCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomLeftCount)
    featuresList.append(topRightCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomRightCount)
    featuresList.append(topRightCount/bottomLeftCount)
    #get center of mass and horizontal histogram
    xCenter, yCenter,xHistogram =histogramAndCenterOfMass(img)
    featuresList.append(xCenter)
    featuresList.append(yCenter)
    #featuresList.extend(xHistogram)
    #print(len(featuresList))
    return featuresList



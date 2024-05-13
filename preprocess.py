from commonfunctions import *
import cv2
import skimage.filters as filters
import scipy.ndimage as nd



def preprocess(img):

    # first apply median to remove s&p noise 
    img = filters.median(img)
    img = rgb2gray(img)
    # Apply median filter to remove salt and pepper noise
    # convert to binary image
    thresh = filters.threshold_otsu(img)
    # print(thresh)
    img = img < thresh
    # img = img.astype(int)
    # cntNotBlack = cv2.countNonZero(img)
    cntNotBlack = np.sum(img!=0)
    # get pixel count of image
    height, width = img.shape
    cntPixels = height*width
    cntBlack = cntPixels - cntNotBlack
    # Ensure the image is white text on Black background because black pixels should be more than white pixels
    if cntBlack < cntNotBlack:
        # img = cv2.bitwise_not(img)
        # take bitwise not using numpy
        img = np.invert(img)

    return img


def removeSkew(img,resizeFactor=0):
    def find_score(arr, angle):
        data = nd.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        # score is the sum of squares of differences of every two consecutive elements in the histogram (variance could also be used)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score


    angles = [0, 45, 90, 135, 180] # TODO 180 might be removed later
    scores = []

    for angle in angles:
        _, score = find_score(img, angle)
        # print('Angle: {}, Score: {}'.format(angle, score))
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.format(best_angle)) # correct skew
    rotated = nd.rotate(img, best_angle, order=0)
    # return rotated
    horizontal_hist = np.sum(img,axis=1,keepdims=True)
    center=np.argmax(horizontal_hist)
    # print('center:',center)
    #plot the histogram to see the peaks
    # cutoff= 590
    # # 590 = 1180/2 and we add 1 to make all the sizes equal for HoG feature extraction
    # if center <=cutoff:
    #     # print('center less than cutoff (fel awl)')
    #     img=rotated[0:cutoff*2+1,0:cutoff*2+1]
    # elif center > img.shape[1]-cutoff:
    #     # print('center more than cutoff (fel a5er)')
    #     img=rotated[-1 -cutoff*2:, -1 -cutoff*2:]
    # else:
    #     # print('center (fel nos)')
    #     img=rotated[center-cutoff-1:center+cutoff,center-cutoff-1:center+cutoff]
    if resizeFactor:
        #convert img to uint8
        rotated = rotated.astype(np.uint8)
        img = cv2.resize(rotated, (resizeFactor, resizeFactor))
    return img

def cropImage(img,cutoff=250):
    # center=img.shape[1]//2
    # make the center dynamic according to highest peak in horizontal histogram
    horizontal_hist = np.sum(img,axis=1,keepdims=True)
    center=np.argmax(horizontal_hist)
    #plot the histogram to see the peaks
    if center <cutoff:
        img=img[0:cutoff*2,0:cutoff*2]
    elif center > img.shape[1]-cutoff:
        img=img[-cutoff*2:, -cutoff*2:]
    else:
        img=img[center-cutoff:center+cutoff,center-cutoff:center+cutoff]
    return img
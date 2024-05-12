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


def removeSkew(img):
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
    return rotated

def cropImage(img):
    center=img.shape[1]//2
    # make the center dynamic according to highest peak in horizontal histogram
    horizontal_hist = np.sum(img,axis=1,keepdims=True)
    center=np.argmax(horizontal_hist)
    #plot the histogram to see the peaks
    cutoff= 250
    if center <cutoff:
        img=img[0:cutoff*2,0:cutoff*2]
    elif center > img.shape[1]-cutoff:
        img=img[-cutoff*2:, -cutoff*2:]
    else:
        img=img[center-cutoff:center+cutoff,center-cutoff:center+cutoff]
    return img
from commonfunctions import *
import cv2
import skimage.filters as filters
import scipy.ndimage as nd
from skimage.transform import rescale, resize


def preprocess(img,first_median_thresh=250_000_000,second_median_thresh=200_000_000):

    # first apply median to remove s&p noise 
    medianimg = filters.median(img)
    # if difference between image after median is large then accept the median image
    if np.sum(np.abs(img-medianimg))>first_median_thresh:
        img=medianimg
        img = filters.median(img)
        if np.sum(np.abs(img-medianimg))>second_median_thresh:
            img=medianimg
    
    
    # determine if image is blurry or not 
    # if blurry apply gaussian filter
    # good value for laplacian is 150
    # if cv2.Laplacian(img, cv2.CV_64F).var() < 150:
    #     img= filters.gaussian(img, sigma=0.1)
    # if image has 3 channels convert to gray
    if len(img.shape) == 3:
        img = rgb2gray(img)
    # convert to binary image
    thresh = filters.threshold_otsu(img)
    # # print(thresh)
    img=np.where(img > thresh, 1, 0)
    cntNotBlack = np.sum(img!=0)
    # get pixel count of image
    height, width = img.shape
    cntPixels = height*width
    cntBlack = cntPixels - cntNotBlack
    # Ensure the image is white text on Black background because black pixels should be more than white pixels
    if cntBlack < cntNotBlack:
        img=1-img

    # # map the original image to itself if binimage is white else 0
    # img= np.where(binimg, img, 0)
    # print(np.max(img))
    # print(np.min(img))
    
    return img


def removeSkew(img,resizeFactor=0):
    def find_score(arr, angle):
        data = nd.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        # score is the sum of squares of differences of every two consecutive elements in the histogram (variance could also be used)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score


    angles = [0, 45, 90, 135] # TODO 180 might be removed later
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
    # horizontal_hist = np.sum(img,axis=1,keepdims=True)
    # print('center:',center)
    # horizontal_hist = np.sum(rotated,axis=1,keepdims=True)
    # # plot the histogram
    # # determine where to cut the image if the histogram dies after some point
    # nonzero=np.nonzero(horizontal_hist)
    # first_nonzero=nonzero[0][0]
    # last_nonzero=nonzero[0][-1]
    # # rotated=rotated[first_nonzero:last_nonzero,:]

    if resizeFactor:
        #convert img to uint8
        # rotated = rotated.astype(np.uint8)
        #skimage resize
        # rotated = resize(rotated, (resizeFactor, resizeFactor))
        factor=resizeFactor/rotated.shape[0]
        rotated = rescale(rotated, factor, anti_aliasing=False)
        # convert to binary image
        rotated = np.where(rotated !=0, 1, 0)
    return rotated

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
from skimage.feature import hog

def extract_hog_features(image):
    # Convert image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    features = hog(image, orientations=9, pixels_per_cell=(16, 8), cells_per_block=(2,2)) #orig was cells_per_block=(2,2)
    return features
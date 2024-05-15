import skimage as ski


def extract_lbp_features(img):
    return ski.feature.local_binary_pattern(img, P=8, R=1)  
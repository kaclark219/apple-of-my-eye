import cv2, numpy as np
from skimage.feature import hog

def extract_features(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hists = []
    for i in range(3):
        hist = cv2.calcHist([img],[i],None,[16],[0,256])
        hist = cv2.normalize(hist,hist).flatten()
        hists.append(hist)
    color_hist = np.hstack(hists)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(10,10), cells_per_block=(2,2), orientations=9, feature_vector=True)

    return np.hstack([color_hist, hog_feat])
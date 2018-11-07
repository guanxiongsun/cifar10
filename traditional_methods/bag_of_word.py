import cv2
from sklearn.feature_extraction import image
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
import time
from utils import load_data as loader

# Get cifar10 dataset
input_shape, x_train, x_test, y_train, y_test = loader.get_cifar10()


def score(clf, X, Y):
    # Pipleline : Extract_patches, Kmeans, SVM
    clf.fit(X, Y)
    predictions = clf.predict(x_test[0:testsamples])
    print("ACC : {}".format(accuracy_score(y_test[0:testsamples], predictions)))
    return


def rgb2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


class PatchExtractor:
    def __init__(self, patch_size=(8, 8)):
        self.patch_size = patch_size

    def _extract_patches(self, x):
        """ Extracts patches from given H x W image """
        # Extract 4x4 patches(8,8) from images
        patches = image.extract_patches(x, self.patch_size, self.patch_size[0])
        patches_vec = patches.reshape((-1, self.patch_size[0] * self.patch_size[1]))
        return patches_vec

    def fit(self, X, Y=None):
        patches = np.concatenate([self._extract_patches(rgb2gray(x)) for x in X])
        print(patches.shape)
        return self

    def transform(self, X, Y=None):
        """ RGB to gray, and do extraction """
        patches = np.array([self._extract_patches(rgb2gray(x)) for x in X])
        # print (patches.shape)
        return patches


class SiftExtractor:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def _extract_sift(self, x):
        """ Extracts SIFT descriptor from image """

        keypoints, descriptors = self.sift_object.detectAndCompute(x, None)
        # print (descriptors.shape)
        return descriptors

    def fit(self, X, Y=None):
        patches = np.concatenate([self._extract_sift(rgb2gray(x)) for x in X])

        print(patches.shape)
        return self

    def transform(self, X, Y=None):
        """ RGB to gray, and do extraction """
        patches = np.array([self._extract_sift(rgb2gray(x)) for x in X])
        # print (patches.shape)
        return patches


class Codebook:
    def __init__(self, size=10):
        self.size = size
        self.clusterer = KMeans(n_clusters=size, verbose=False)

    def _get_histogram(self, x):
        """ Returns histogram of codewords for given features """
        # Assign each patch to a cluster
        clusters = self.clusterer.predict(x)
        # Get the number of each patch type
        return np.bincount(clusters, minlength=self.size)

    def fit(self, X, Y=None):
        """ Fitting Kmeans """
        print("Fitting cluster ... waiting ...")
        # print(np.concatenate(X).shape)
        start = time.clock()
        self.clusterer.fit(np.concatenate(X))
        elapsed = time.clock() - start
        print ('Kmeans used: %.2f s' % elapsed)
        return self

    def transform(self, X, Y=None):
        return np.array([self._get_histogram(x) for x in X])


# Number of samples to train [0, 50000]
nsamples = 500
# Number of samples to test [0, 10000]
testsamples = 10000
X = x_train[0:nsamples]
Y = y_train[0:nsamples]
Y = Y.reshape(-1)
patcher = PatchExtractor(patch_size=(8, 8))
sift = SiftExtractor()
codebook = Codebook(size=50)
clf = svm.SVC(kernel='rbf')
pipeline = Pipeline([("Feature_extractor", patcher), ("Codebook", codebook), ("svm", clf)])
score(pipeline, X, Y)

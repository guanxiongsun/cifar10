from sklearn.feature_extraction import image
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import time
from utils import load_data as loader
import fisher_vectors as fv
import cv2

# Get cifar10 dataset
input_shape, x_train, x_test, y_train, y_test = loader.get_cifar10()


def score(clf, X, Y):
    # Pipleline : Extract_patches, Kmeans, SVM
    clf.fit(X, Y)
    predictions = clf.predict(x_test[0:testsamples])
    print("ACC : {}".format(accuracy_score(y_test[0:testsamples], predictions)))
    return


def rgb2gray(img):
    # print (img.shape)
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
        if descriptors is not None:
            return descriptors
        else:
            print ("None descriptor!")
            return np.zeros([1, 128])

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        """ RGB to gray, and do extraction """
        patches = np.array([self._extract_sift(rgb2gray(x)) for x in X])
        print (patches.shape)
        return patches


class Codebook:
    def __init__(self, size=10):
        self.size = size
        self.clusterer = KMeans(n_clusters=size, verbose=False)

    def _get_histogram(self, x):
        """ Returns histogram of codewords for given features
            x.shape = [n_descriptor, dscpt_dimension]
        """
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


class FisherVectorBook:
    def __init__(self, size=64):
        self.size = size
        self.clusterer = GaussianMixture(n_components=size, covariance_type='diag')

    def normalize(self, fv):
        v = np.sqrt(abs(fv)) * np.sign(fv)
        return v / np.sqrt(np.dot(v, v))

    def _get_predict(self, x):
        """ Returns histogram of codewords for given features
            x.shape = [n_descriptor, dscpt_dimension]
        """
        fisher_vec = fv.fisher_features(x, self.clusterer)
        print (fisher_vec.shape)
        return self.normalize(fisher_vec)

    def fit(self, X, Y=None):
        """ Fitting Kmeans """
        print("Fitting GMM ... waiting ...")
        # print(np.concatenate(X).shape)
        start = time.clock()
        self.clusterer.fit(np.concatenate(X))
        elapsed = time.clock() - start
        print ('GMM used: %.2f s' % elapsed)
        return self

    def transform(self, X, Y=None):
        print (X.shape[0])
        return np.array([self._get_predict(x) for x in X])


# Number of samples to train [0, 50000]
nsamples = 50000
# Number of samples to test [0, 10000]
testsamples = 10000
X = x_train[0:nsamples]
Y = y_train[0:nsamples]
Y = Y.reshape(-1)

patcher = PatchExtractor(patch_size=(8, 8))
sift = SiftExtractor()
fisher = FisherVectorBook(size=64)
codebook = Codebook(size=64)
pca = PCA(n_components=64)
clf = svm.SVC(kernel='rbf', verbose=False)

pipeline = Pipeline([("Feature_extractor", sift),
                     ("Codebook", codebook),
                     # ("PCA", pca),
                     ("svm", clf)])
score(pipeline, X, Y)

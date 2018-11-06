import sklearn
import cv2
from sklearn.feature_extraction import image
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
import time
import load_data

input_shape, x_train, x_test, y_train, y_test = load_data.get_cifar10()


# def score(clf, X, Y, folds=2, verbose=False):
#     predictions = np.zeros(len(Y))
#     kf = KFold(n_splits=folds, shuffle=True)
#
#     for train, test in kf.split(X):
#         clf.fit(X[train], Y[train])
#         predictions[test] = clf.predict(X[test])
#         if verbose:
#             print("Fold : {}".format(accuracy_score(Y[test], predictions[test])))
#         #     for i, (train, test) in enumerate(KFold(len(X), n_folds=folds, shuffle=True)):
#     #         clf.fit(X[train], Y[train])
#     #         predictions[test] = clf.predict(X[test])
#     #         if verbose:
#     #             print("Fold {}: {}".format(i + 1, accuracy_score(Y[test], predictions[test])))
#     return accuracy_score(Y, predictions)


def score(clf, X, Y, verbose=False):
    # Pipleline : Extract_patches, Kmeans, SVM
    clf.fit(X, Y)
    predictions = clf.predict(x_test)
    if verbose:
        print("ACC : {}".format(accuracy_score(y_test, predictions)))
    return accuracy_score(Y, predictions)


def rgb2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


class PatchExtractor(sklearn.base.BaseEstimator):
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


class Codebook():
    def __init__(self, size=10):
        self.size = size
        self.clusterer = KMeans(n_clusters=size, verbose=True)

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


X = x_train
Y = y_train
patcher = PatchExtractor(patch_size=(8, 8))
codebook = Codebook(size=128)
clf = svm.SVC(kernel='linear')
pipeline = Pipeline([("Patch_extractor", patcher), ("Codebook", codebook), ("svm", clf)])
score(pipeline, X, Y, verbose=True)

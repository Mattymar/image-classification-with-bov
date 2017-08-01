import numpy as np
import cv2
import pandas as pd
import os
import cluster
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.decomposition import PCA


class RootSIFT:
    # Derived from 'http://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/'
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        kps, descs = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return and empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        return kps, descs


class MacOSFile(object):
    """Used to work around memory issues for large pickled files on MacOS"""

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def get_root_sift_features(labeled_paths):
    """Get feature descriptors for each image"""
    
    rs = RootSIFT()
    detector = cv2.xfeatures2d.SIFT_create()
    train_descs_list = []
    train_img_dicts_list = []
    test_img_dicts_list = []
    train_paths, test_paths = train_test_split(labeled_paths, test_size=0.3)

    for path, label in train_paths:
        try:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect SIFT keypoints
            kps = detector.detect(gray, None)

            # Detect MSER keypoints
            mser = cv2.MSER_create()
            mser_kps = mser.detect(gray)

            # Calculate descriptors for SIFT and MSER keypoints and stack them
            kps, sift_descs = rs.compute(gray, kps)
            kps, mser_descs = rs.compute(gray, mser_kps)

            img_descs = []
            img_descs.append(mser_descs)
            img_descs.append(sift_descs)
            img_descs = np.vstack(img_descs)

            # ensure each descriptor is a 1 x 128 vector
            if img_descs.shape[1] == 128:
                train_descs_list.append(sift_descs)
                train_descs_list.append(mser_descs)

                img_dict = {'descriptors': img_descs, 'label': label}
                train_img_dicts_list.append(img_dict)
            else:
                print(sift_descs.shape, mser_descs.shape)

        except:
            # Skip and note any failed images
            print(path + ' Could not be converted')

    # Repeat above process for test images
    # TODO: Add function to eliminate repeated code
    for path, label in test_paths:
        try:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kps = detector.detect(gray, None)

            mser = cv2.MSER_create()
            mser_kps = mser.detect(gray)

            kps, sift_descs = rs.compute(gray, kps)
            kps, mser_descs = rs.compute(gray, mser_kps)

            img_descs = []
            img_descs.append(mser_descs)
            img_descs.append(sift_descs)
            img_descs = np.vstack(img_descs)

            if img_descs.shape[1] == 128:
                img_dict = {'descriptors': img_descs, 'label': label}
                test_img_dicts_list.append(img_dict)
        except:
            print(path + ' Could not be converted')
    train_descs_array = np.vstack(train_descs_list)

    # Store feature vectors as pickled files
    pickle_dump(train_descs_array, 'all_descriptors_full_10c.pkl')
    pickle_dump(train_img_dicts_list, 'train_img_dicts_list_full_10c.pkl')
    pickle_dump(test_img_dicts_list, 'test_img_dicts_list_full_10c.pkl')

    return train_descs_array, train_img_dicts_list, test_img_dicts_list


def get_image_labels_and_paths(images_path):
    """
    :param images_path: The path to the directory holding image categories
    :return: image_path, label tuple for each image
    """
    image_labels_and_paths = []
    for category in os.listdir(images_path):
        if not category.startswith('.'):

            for f in os.listdir(images_path + '/' + category):
                image_labels_and_paths.append((os.path.join(images_path + '/' + category, f), category))
    return image_labels_and_paths


def spm_split(image):
    """Splits image into grids of various sizes for spatial tests"""
    
    h = image.shape[0]
    w = image.shape[1]

    whole = image[:]

    q1 = image[0:h//2, 0:w//2]
    q2 = image[0: h//2, w//2:]
    q3 = image[h//2:, w//2:]
    q4 = image[h//2:, 0:w//2]

    s1 = image[0:h//4, 0:w//4]
    s2 = image[h//4:h//4 * 2, 0:w//4]
    s3 = image[h//4*2:h//4*3, 0:w//4]
    s4 = image[h//4*3:, 0:w//4]

    s5 = image[0:h // 4, w//4:w//4 * 2]
    s6 = image[h // 4:h // 4 * 2, w//4:w//4 * 2]
    s7 = image[h // 4 * 2:h // 4 * 3, w//4:w//4 * 2]
    s8 = image[h // 4 * 3:, w//4:w//4 * 2]

    s9 = image[0:h // 4, w // 4 * 2:w // 4 * 3]
    s10 = image[h // 4:h // 4 * 2, w // 4 * 2:w // 4 * 3]
    s11 = image[h // 4 * 2:h // 4 * 3, w // 4 * 2:w // 4 * 3]
    s12 = image[h // 4 * 3:, w // 4 * 2:w // 4 * 3]

    s13 = image[0:h // 4, w // 4 * 3:]
    s14 = image[h // 4:h // 4 * 2, w // 4 * 3:]
    s15 = image[h // 4 * 2:h // 4 * 3, w // 4 * 3:]
    s16 = image[h // 4 * 3:, w // 4 * 3:]

    return whole, q1, q2, q3, q4, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16


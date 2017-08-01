import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle
import sift


def build_codebook(all_features_array):
    """Build the feature space for all images in the training set"""
    all_features_array = sift.pickle_load(all_features_array)
    nfeatures = all_features_array.shape[0]
    nclusters = int(np.sqrt(nfeatures) * 2)
    codebook = MiniBatchKMeans(n_clusters=nclusters, max_iter=500).fit(all_features_array.astype('float64'))

    return codebook


def build_codebook_spm(all_features_dict):
    """Build feature space for each spatial level in all images of training set"""
    
    with open(all_features_dict, 'rb') as all_features_dict:
        all_features_dict = pickle.load(all_features_dict)
        levels = ['l0', 'l1', 'l2']
        levels_codebook = {}
        for level in levels:
            nfeatures = all_features_dict[level].shape[0]
            nclusters = 200#int(np.sqrt(nfeatures) * 2)
            levels_codebook[level] = MiniBatchKMeans(n_clusters=nclusters, max_iter=500).fit(all_features_dict[level].astype('float64'))

        return levels_codebook


def build_image_histograms(codebook, train_img_dicts_list, test_img_dicts_list):
    """Build the feature vector for each image"""
    
    print(codebook.n_clusters)
    s = list(range(codebook.n_clusters))
    s.append('y')
    train_df = pd.DataFrame(index=s)
    test_df = pd.DataFrame(index=s)

    train_img_dicts_list = sift.pickle_load(train_img_dicts_list)
    for i in range(len(train_img_dicts_list)):
        try:
            # Match each descriptor vector to a cluster in the codebook
            codewords = [codebook.predict(desc.reshape(1, -1))[0] for desc in train_img_dicts_list[i]['descriptors']]
        except:
            print(i)

        # Build a histogram of number of ocurrences of all k features for each image
        histogram, clusters = np.histogram(codewords, bins=range(0, codebook.n_clusters + 1), density=False)
        data = list(histogram)
        data.append(train_img_dicts_list[i]['label'])

        train_df[i] = data

    test_img_dicts_list = sift.pickle_load(test_img_dicts_list)
    for i in range(len(test_img_dicts_list)):
        try:
            codewords = [codebook.predict(desc.reshape(1, -1))[0] for desc in test_img_dicts_list[i]['descriptors']]
        except:
            print(i)
        histogram, clusters = np.histogram(codewords, bins=range(0, codebook.n_clusters + 1), density=False)
        data = list(histogram)
        data.append(test_img_dicts_list[i]['label'])

        test_df[i] = data

    sift.pickle_dump(train_df, 'train_df_full_10c.pkl')
    sift.pickle_dump(test_df, 'test_df_full_10c.pkl')

    return train_df, test_df


def build_image_histograms_spm(levels_codebook, train_img_dicts_list, test_img_dicts_list):
    """Build a feature vector for each spatial level of an image"""
    
    s = list(range(codebook.n_clusters))
    s.append('y')
    train_df = pd.DataFrame(index=s)
    test_df = pd.DataFrame(index=s)

    train_img_dicts_list = sift.pickle_load(train_img_dicts_list)
    for i in range(len(train_img_dicts_list)):
        try:
            # Match each descriptor vector to a cluster in the codebook
            codewords = [codebook.predict(desc.reshape(1, -1))[0] for desc in train_img_dicts_list[i]['descriptors']]
        except:
            print(i)

        # Build a histogram of number of ocurrences of all k features for each image
        histogram, clusters = np.histogram(codewords, bins=range(0, codebook.n_clusters + 1), density=False)
        data = list(histogram)
        data.append(train_img_dicts_list[i]['label'])

        train_df[i] = data

    test_img_dicts_list = sift.pickle_load(test_img_dicts_list)
    for i in range(len(test_img_dicts_list)):
        try:
            codewords = [codebook.predict(desc.reshape(1, -1))[0] for desc in test_img_dicts_list[i]['descriptors']]
        except:
            print(i)
        histogram, clusters = np.histogram(codewords, bins=range(0, codebook.n_clusters + 1), density=False)
        data = list(histogram)
        data.append(test_img_dicts_list[i]['label'])

        test_df[i] = data

    with open('train_df_4000_8cx.pkl', 'wb') as trdf:
        pickle.dump(train_df, trdf)

    with open('test_df_4000_8cx.pkl', 'wb') as tedf:
        pickle.dump(test_df, tedf)

    return train_df, test_df

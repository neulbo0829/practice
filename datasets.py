import numpy as np
from feature_extraction import *
import json

# get filenames to use in json file
train_list = get_filenames_without_extension('./train')
test_list = get_filenames_without_extension('./test')

# load text captions
with open('./osa_STOPBANG_mapped.json', 'r') as f:
    data = json.load(f)

def load_dataset(load_test = False):
    
    if load_test:
        # captions
        test_caps = []
        for file_name in test_list:
            # connect each values with ','
            joined_value = ', '.join(data[file_name].values())
            test_caps.append(joined_value)
        # image features
        test_ims = np.load('test.npy')
        return (test_caps, test_ims)
    else: # add dev part if it is needed
        # captions
        train_caps = []
        for file_name in train_list:
            # connect each values with ','
            joined_value = ', '.join(data[file_name].values())
            train_caps.append(joined_value)
        # image features
        train_ims = np.load('train.npy')
        return (train_caps, train_ims)


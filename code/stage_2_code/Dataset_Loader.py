'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []

        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
            for line in f:
                elements = line.strip().split()

                # try to safely parse
                try:
                    elements = [int(i) for i in elements]
                except:
                    continue

                # skip only if clearly invalid
                if len(elements) < 2:
                    continue

                # label first, rest features
                y.append(elements[0])
                X.append(elements[1:])

        print("Loaded X shape:", np.array(X).shape)
        return {'X': X, 'y': y}

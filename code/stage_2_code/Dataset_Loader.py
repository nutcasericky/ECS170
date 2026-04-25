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
        import os
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        print("Opening file:", path)
        print("File exists?", os.path.exists(path))
        print("Absolute path:", os.path.abspath(path))
        # Above is debugging b/c I couldn't figure out where to open file
        print('loading data...')

        X = []
        y = []

        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                elements = line.strip().replace(',', ' ').split()

                cleaned = []
                for i in elements:
                    i = i.strip()
                    if i == '':
                        continue
                    try:
                        cleaned.append(int(i))
                    except:
                        continue

                elements = cleaned

                if len(elements) < 10:
                    continue

                # Used ChatGPT to help explain the continue lines.
                y.append(elements[0])
                X.append(elements[1:])

        print("Number of samples:", len(X))
        if len(X) > 0:
            print("Loaded X shape:", np.array(X).shape)
            print("First sample length:", len(X[0]))

        return {'X': X, 'y': y}

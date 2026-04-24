'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Had to search up what to import

class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        results = {
            "Accuracy": accuracy_score(true_y, pred_y), # Calculate Accuracy
            "Precision_macro": precision_score(true_y, pred_y, average='macro'), # Calculate Precision (macro)
            "Recall_macro": recall_score(true_y, pred_y, average='macro'), # Calculate Recall (macro)
            "F1_macro": f1_score(true_y, pred_y, average='macro') # Calculate F1 (macro)
        }
        # Had to search up how to find Precision, Recall, and F1, also what "macro" was.

        print(results)

        return results["Accuracy"]  # keeping this so KFold still works


# Original Code
"""class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y']) This is what line 33 is based on
"""

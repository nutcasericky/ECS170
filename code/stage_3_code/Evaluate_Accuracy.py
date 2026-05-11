from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy:
    def __init__(self, name="cnn evaluation", description=""):
        self.name = name
        self.description = description
        self.data = None

    def evaluate(self):
        print("evaluating performance...")

        true_y = self.data["true_y"]
        pred_y = self.data["pred_y"]

        results = {
            "Accuracy": accuracy_score(true_y, pred_y),
            "Precision_macro": precision_score(true_y, pred_y, average="macro", zero_division=0),
            "Recall_macro": recall_score(true_y, pred_y, average="macro", zero_division=0),
            "F1_macro": f1_score(true_y, pred_y, average="macro", zero_division=0)
        }

        print("Accuracy:", results["Accuracy"])
        print("Precision_macro:", results["Precision_macro"])
        print("Recall_macro:", results["Recall_macro"])
        print("F1_macro:", results["F1_macro"])

        return results

from stage_3_code.Dataset_Loader import Dataset
from stage_3_code.Method_CNN import CNNMethod
import numpy as np
import torch


# ---- CNN script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    dataset_names = ["ORL", "MNIST", "CIFAR"]
    # ------------------------------------------------------

    for dataset_name in dataset_names:

        # ---- object initialization section ---------------
        data_obj = Dataset(
            dataset_name=dataset_name,
            data_dir="../../data/stage_3_data"
        )

        method_obj = CNNMethod(
            dataset_name=dataset_name,
            epochs=10,
            batch_size=64,
            learning_rate=0.001
        )
        # --------------------------------------------------

        # ---- running section -----------------------------
        print("************ Start ************")
        print("Dataset:", dataset_name)

        X_train, y_train, X_test, y_test = data_obj.load()

        accuracy = method_obj.run(
            X_train,
            y_train,
            X_test,
            y_test
        )

        print("************ Overall Performance ************")
        print(dataset_name + " CNN Accuracy: " + str(accuracy))
        print("************ Finish ************")
        print()
        # --------------------------------------------------
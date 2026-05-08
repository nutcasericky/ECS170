from stage_3_code.Dataset_Loader import Dataset
from stage_3_code.Method_CNN import CNNMethod

dataset_names = ["ORL", "MNIST", "CIFAR"]

for dataset_name in dataset_names:
    print("Running dataset:", dataset_name)

    dataset = Dataset(
        dataset_name=dataset_name,
        data_dir="../../data/stage_3_data"
    )

    X_train, y_train, X_test, y_test = dataset.load()

    method = CNNMethod(
        dataset_name=dataset_name,
        epochs=10,
        batch_size=64,
        learning_rate=0.001
    )

    accuracy = method.run(X_train, y_train, X_test, y_test)

    print(dataset_name, "Final Accuracy:", accuracy)
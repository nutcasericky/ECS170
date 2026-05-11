import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


class CNNMethod:
    def __init__(self, dataset_name="ORL", epochs=10, batch_size=64, learning_rate=0.001,
                 result_dir="../../result/stage_3_result"):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.result_dir = result_dir
        self.model = None
        self.evaluate_obj = Evaluate_Accuracy("cnn evaluation", "")
        self.final_results = None

    def run(self, trainData, trainLabel, testData, testLabel):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainData = torch.tensor(trainData, dtype=torch.float32)
        trainLabel = torch.tensor(trainLabel, dtype=torch.long)
        testData = torch.tensor(testData, dtype=torch.float32)
        testLabel = torch.tensor(testLabel, dtype=torch.long)

        train_dataset = TensorDataset(trainData, trainLabel)
        test_dataset = TensorDataset(testData, testLabel)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        in_channels = trainData.shape[1]
        input_height = trainData.shape[2]
        input_width = trainData.shape[3]
        num_classes = len(np.unique(trainLabel.numpy()))

        self.model = CNNModel(in_channels, num_classes, input_height, input_width).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_losses = []
        test_accuracies = []

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            results = self.evaluate(test_loader, device)
            accuracy = results["Accuracy"]
            test_accuracies.append(accuracy)

            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

        self.plot_curves(train_losses, test_accuracies)
        return self.final_results

    def evaluate(self, test_loader, device):
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = self.model(X_batch)
                predicted = torch.argmax(outputs, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        self.evaluate_obj.data = {
            "true_y": all_labels,
            "pred_y": all_preds
        }

        results = self.evaluate_obj.evaluate()
        self.final_results = results

        return results

    def plot_curves(self, train_losses, test_accuracies):
        os.makedirs(self.result_dir, exist_ok=True)
        plt.figure()
        plt.plot(train_losses)
        plt.title(f"{self.dataset_name} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.result_dir, f"{self.dataset_name}_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(test_accuracies)
        plt.title(f"{self.dataset_name} Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(self.result_dir, f"{self.dataset_name}_accuracy.png"))
        plt.close()


class CNNModel(nn.Module):
    def __init__(self, in_channels, num_classes, input_height, input_width):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._to_linear = None
        self._get_conv_output(in_channels, input_height, input_width)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, in_channels, input_height, input_width):
        x = torch.randn(1, in_channels, input_height, input_width)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

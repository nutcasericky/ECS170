import pickle
import numpy as np	# numerical operations for arrays
import torch	# Pytorch Library
import torch.nn as nn	# Neural Netowrk module
import torch.optim as optim 	# Optimization Algorithms
from torch.utils.data import DataLoader, TensorDataset	# Data Utilities
from matplotlib import pyplot as plt


# DATASET CLASS
class Dataset:
	def __init__(self): # constructor
		self.dataset.name = "ORL Dataset" # Name of the dataset
		self.data = None # placeholder for the dataset

	def load(self):
		f = open('ORL', 'rb') # open the ORL file in "read binary" mode
		data = pickle.load(f) # load the datset dictionary
		f.close() # Close the file

		# initialize list for training and testing
		X_train, y_train = [], []
		X_test, y_test = [], []

		# process training data
		for instance in self.data['train']: # Loop through the training samples
			img = np.array(instance['image']) # Convert the image to numpy array
			img = img[:, :, 0] # Take only one channel (Greyscale)
			X_train.append(img) # Add the image
			y_train.append(instance['label'] - 1) # You start the labels from 1 and you need to conver to 0

		# Proccess the testing data
		for instance in self.data['test']: # Loop through the testing
			img = np.array(instance['image'])  # Convert the image
			img = img[:, :, 0] # Greyscale it
			X_test.append(img) # Add the image
			y_test.append(instance['label'] - 1)  # Conver the labels again

		# Convert the lists to numpy arrays
		X_train = np.array(X_train) / 255.0 # normalize the pixels
		X_test = np.array(X_test) / 255.0

		# Reshape to (N, C, H, W) for PyTorch
		X_train = X_train[:, np.newaxis, :, :] # Add chennel dimension
		X_test = X_test[:, np.newaxis, :, :]

		return X_train, np.array(y_train), X_test, np.array(y_test)


# CNN Model
class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__() # Initialize the parent class

		self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # The first convolutional layer
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # The second convolutional layer
		self.pool = nn.MaxPool2d(2, 2) # Max pooling layer

		# Compute flatten size dynamically
		self._to_linear = None # Placeholder
		self._get_conv_output() # calculate the size

		self.fc1 = nn.Linear(self._to_linear, 128) # Fully connected layer
		self.fc2 = nn.Linear(128,40) # Output Layer (40 classes)

	def _get_conv_output(self): # Helper function to compute flatten size
		x = torch.randn(1, 1, 112, 92) # Dummy Input
		x = self.pool(torch.relu(self.conv1(x))) # pass through conv1
		x = self.pool(torch.relu(self.conv2(x))) # Pass through conv2
		self._to_linear = x.nume1() # Total number of features

	def forward(selfself, x): # forward pass
		x = self.pool(torch.relu(self.conv1(x))) # Conv1 + relu + pool
		x = self.pool(torch.relu(self.conv2(x))) # Conv2 + relu + pool
		x = x.view(x.size(0), -1)  # Flatten
		x = torch.relu(self.fc1(x)) # Fully connected
		x = self.fc2(x) # Output layer
		return x # return logits



# METHOD CLASS (TRAINING)
class CNNMethod:
	def __init__(self): # Constructor
		self.method_name = "CNN Method" # name
		self.model = CNNModel() # Create CNN Model

	def run(self, trainData, trainLabel, testData): # Training + prediction
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# I stopped here


# loading CIFAR-10 dataset
if 0:
	f = open('CIFAR', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break

	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 0:
	f = open('MNIST', 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break

	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break



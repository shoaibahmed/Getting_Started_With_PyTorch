import os
import sys
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

PYTHON_3 = True
if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle
else:
	print ("Using Python 2")
	import cPickle as pickle
	PYTHON_3 = False

class MyDataset(Dataset):
	"""My custom dataset."""

	def __init__(self, root_dir, split, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the pickle files.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.files = [os.path.abspath(os.path.join(root_dir, x)) for x in os.listdir(root_dir)]
		self.transform = transform
		self.data = None
		self.labels = np.array([])
		for file in self.files:
			with open(file, 'rb') as inputFile:
				if PYTHON_3:
					data = pickle.load(inputFile, encoding='latin1')
				else:
					data = pickle.load(inputFile)

				if self.data is None:
					self.data = data[0]
				else:
					self.data = np.vstack((self.data, data[0]))

				self.labels = np.concatenate((self.labels, data[1]))

		# Split in train/test
		self.split = split
		if self.split == 'Train':
			pass # Select only training examples (modify self.data and self.labels)
		elif self.split == 'Validation':
			pass # Select only validation examples
		elif self.split == 'Test':
			pass # Select only test examples
		else:
			print ("Error: Unknown data split!")
			exit (-1)

		########### Integrated into the model itself ###########
		# # Reshape the data into sequences (only for LSTM)
		# numFeatures = self.data.shape[1]
		# numSequences = 5
		# sequenceLength = int(numFeatures / numSequences)
		# self.data = np.reshape(self.data, [-1, numSequences, sequenceLength])
		########################################################

		print ("Data shape:", self.data.shape)
		print ("Labels shape:", self.labels.shape)
	
	def getFeatureVectorLength(self):
		return self.data.shape[1]

	def getNumClasses(self):
		return len(np.unique(self.labels))

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sample = {'data': self.data[idx, :], 'label': self.labels[idx]}

		if self.transform:
			sample = self.transform(sample)

		return sample


if __name__ == "__main__":
	print ("Test dataloader")
	dataset = MyDataset("../data/p_data", split="Train")
	dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=5, shuffle=False)

	for idx, data in enumerate(dataLoader):
		X = data["data"]
		y = data["label"]
		print (X.shape)
		print (y.shape)

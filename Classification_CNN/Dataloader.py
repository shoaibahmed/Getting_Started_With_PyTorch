import os
import sys
from random import shuffle
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from enum import Enum


class Data(Enum):
	TRAIN = 1
	TEST = 2


class MyDataset(Dataset):
	"""My custom dataset."""

	def __init__(self, root_dir, split=Data.TRAIN, transform=None, useClassNames=True):
		"""
		Args:
			root_dir (string): Directory with all the pickle files.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.files = []
		self.useClassNames = useClassNames
		self.imageFormats = [".jpg", ".png", ".bmp"]

		if self.useClassNames:
			self.class_names = [cls for cls in os.listdir(root_dir) if cls != "./"]

			# Assign IDs to the class names
			self.class_dict = {}
			for idx, cls in enumerate(self.class_names):
				self.class_dict[cls] = idx
		else:
			self.classes = [int(cls) for cls in os.listdir(root_dir) if cls != "./"]

		for root, dirs, files in os.walk(root_dir):
			for file in files:
				for imageFormat in self.imageFormats:
					if file.endswith(imageFormat):
						self.files.append(os.path.abspath(os.path.join(root, file)))
						break

		# Randomly shuffle the files
		shuffle(self.files)

		self.transform = transform
		print("Classes:", self.class_dict)
		self.oneHot = False # Torch can't deal with one-hot vectors

	def getNumClasses(self):
		return len(self.classes)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file = self.files[idx]
		img = Image.open(file)
		if self.transform:
			img = self.transform(img)

		if self.useClassNames:
			label = self.class_dict[file.split(os.sep)[-3]]
		else:
			label = int(file.split(os.sep)[-3])

		if self.oneHot:
			oneHot = np.zeros(self.getNumClasses())
			oneHot[label] = 1.0
			label = oneHot

		sample = {'data': img, 'label': label}
		return sample


if __name__ == "__main__":
	print("Test dataloader")

	dataTransform = transforms.Compose([
		transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	dataset = MyDataset("../Dataset/ASLLVD", transform=dataTransform)
	dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=5, shuffle=False)

	for idx, data in enumerate(dataLoader):
		X = data["data"]
		y = data["label"]
		print(X.shape)
		print(y.shape)

		if idx == 5:
			break

import os
import sys
import numpy as np
import cv2

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

class SegmentationDataset(Dataset):
	"""Loader for a segmentation dataset."""

	def __init__(self, rootDir, imagesDir="Images", maskDir="SegmentationClass", maskExt=None, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the pickle files.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.rootDir = rootDir
		self.imagesDir = os.path.abspath(os.path.join(rootDir, imagesDir))
		self.maskDir = os.path.abspath(os.path.join(rootDir, maskDir))

		# Verify path exists
		if not os.path.exists(self.imagesDir):
			print ("Error: Images directory does not exist (%s)" % self.imagesDir)
		if not os.path.exists(self.maskDir):
			print ("Error: Mask directory does not exist (%s)" % self.maskDir)

		self.images = [x for x in os.listdir(self.imagesDir)]
		self.transform = transform
		self.maskExt = maskExt
		print ("Number of files found:", len(self.images))
	
	def getNumClasses(self):
		return 3 # Background, Row, Col

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		# Load the corresponding image and the mask
		img = cv2.imread(os.path.join(self.imagesDir, self.images[idx]), cv2.IMREAD_COLOR)
		maskFileName = self.images[idx][:self.images[idx].rfind('.')]
		if self.maskExt is None:
			maskFileExt = self.images[idx][self.images[idx].rfind('.'):]
		else:
			maskFileExt = self.maskExt
		rowMaskFileName = os.path.join(self.maskDir, maskFileName + "-row" + maskFileExt)
		colMaskFileName = os.path.join(self.maskDir, maskFileName + "-column" + maskFileExt)
		maskRow = cv2.imread(rowMaskFileName, cv2.IMREAD_GRAYSCALE)
		maskCol = cv2.imread(colMaskFileName, cv2.IMREAD_GRAYSCALE)
		sample = {'image': img, 'mask': np.stack((maskRow, maskCol), axis=2)}

		if self.transform:
			sample = self.transform(sample)

		return sample


if __name__ == "__main__":
	print ("Test dataloader")
	dataset = SegmentationDataset("./data/", imagesDir="Images", maskDir="SegmentationClass", maskExt=None)
	dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

	for idx, data in enumerate(dataLoader):
		X = data["image"]
		y = data["mask"]
		print (X.shape)
		print (y.shape)

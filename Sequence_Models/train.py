#!/bin/python
from optparse import OptionParser

import os
import sys
import shutil
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Custom imports
from Dataloader import *
from Models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(options):
	# Clear output directory
	if os.path.exists(options.outputDir):
		print ("Removing old directory!")
		shutil.rmtree(options.outputDir)
	os.mkdir(options.outputDir)

	# Create dataloader
	dataset = MyDataset(options.rootDir, split=Data.TRAIN)
	dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=options.batchSize, shuffle=False)

	# Create model
	inputShape = dataset.getDataShape()
	numberOfClasses = dataset.getNumClasses()
	print ("Input shape: %s | Number of classes: %d" % (str(inputShape), numberOfClasses))

	if options.useLSTM:
		featureVectorLength = inputShape[0]
		assert((featureVectorLength % options.sequenceLength) == 0, "Error: Sequence length should split the sequence into equal chunks")
		featureVectorLength = int(featureVectorLength / options.sequenceLength)
		model = LSTM(options.sequenceLength, featureVectorLength, hiddenDims=options.hiddenStateDims, numClasses=numberOfClasses,
					 numLayers=options.numLayers, dropout=0.0, bidirectional=options.bidirectional)
	else:
		model = CNN(numberOfClasses, inputShape, numFilters=(32, 64, 128))

	model.to(device) # Move the model to desired device

	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)

	# Define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

	# Define loss function
	criterion = torch.nn.CrossEntropyLoss()

	for epoch in range(options.trainingEpochs):
		# Start training
		for iterationIdx, data in enumerate(dataLoader):
			X = data["data"]
			y = data["label"]

			# Move the data to PyTorch on the desired device
			X = Variable(X).float().to(device)
			y = Variable(y).long().to(device)

			# Get model predictions
			pred = model(X)

			# Optimize
			optimizer.zero_grad()
			loss = criterion(pred, y)
			loss.backward()
			optimizer.step()

			if iterationIdx % options.displayStep == 0:
				print ("Epoch %d | Iteration: %d | Loss: %.5f" % (epoch, iterationIdx, loss))

		# Save model
		torch.save(model.state_dict(), os.path.join(options.outputDir, "model.pth"))

def test(options):
	# Clear output directory
	if not os.path.exists(options.outputDir):
		print ("Error: Model directory does not exist!")
		exit (-1)

	# Create dataloader
	dataset = MyDataset(options.rootDir, split=Data.TEST)
	dataLoader = DataLoader(dataset=dataset, num_workers=1, batch_size=options.batchSize, shuffle=False)

	# Create model
	inputShape = dataset.getDataShape()
	numberOfClasses = dataset.getNumClasses()
	print("Input shape: %s | Number of classes: %d" % (str(inputShape), numberOfClasses))

	if options.useLSTM:
		featureVectorLength = inputShape[0]
		assert ((featureVectorLength % options.sequenceLength) == 0,
				"Error: Sequence length should split the sequence into equal chunks")
		featureVectorLength = int(featureVectorLength / options.sequenceLength)
		model = LSTM(options.sequenceLength, featureVectorLength, hiddenDims=options.hiddenStateDims, numClasses=numberOfClasses,
					 numLayers=options.numLayers, dropout=0.0, bidirectional=options.bidirectional)
	else:
		model = CNN(numberOfClasses, inputShape, numFilters=(32, 64, 128))

	# Save model
	modelCheckpoint = torch.load(os.path.join(options.outputDir, "model.pth"))
	model.load_state_dict(modelCheckpoint)
	print("Model restored!")

	gtLabels = []
	predictedLabels = []
	for iterationIdx, data in enumerate(dataLoader):
		X = data["data"]
		y = data["label"]

		# Move the data to PyTorch on the desired device
		X = Variable(X).float().to(device)
		y = Variable(y).long().to(device)

		# Get model predictions
		outputs = model(X)

		# Check prediction
		_, preds = torch.max(outputs.data, dim=1)
		correctPred = torch.sum(preds == y.data)
		correctExamples = correctPred.item()

		# Add the labels
		gtLabels.append(data["label"])
		predictedLabels.append(preds.numpy())

		print("Iteration: %d | Correct examples: %d | Total examples: %d | Accuracy: %.5f" % (iterationIdx, correctExamples, len(predictedLabels[-1]), float(correctExamples) / len(predictedLabels[-1])))

	# Compute statistics
	gtLabels = np.array(gtLabels).flatten()
	predictedLabels = np.array(predictedLabels).flatten()

	print("GT labels shape:", gtLabels.shape)
	print("Predicted labels shape:", predictedLabels.shape)


if __name__ == "__main__":
	# Command line options
	parser = OptionParser()

	# Base options
	parser.add_option("-m", "--model", action="store", type="string", dest="model", default="NAS", help="Model to be used for Cross-Layer Pooling")
	parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
	parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
	parser.add_option("-o", "--outputDir", action="store", type="string", dest="outputDir", default="./output", help="Output directory")
	parser.add_option("-e", "--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Number of training epochs")
	parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=10, help="Batch Size")
	parser.add_option("-d", "--displayStep", action="store", type="int", dest="displayStep", default=2, help="Display step where the loss should be displayed")

	# Network params
	parser.add_option("--useLSTM", action="store_true", dest="useLSTM", default=False, help="Use LSTM network instead of a CNN")
	parser.add_option("--bidirectional", action="store_true", dest="bidirectional", default=False, help="Whether to use bidirectional LSTM")
	parser.add_option("--hiddenStateDims", action="store", type="int", dest="hiddenStateDims", default=10, help="Dimensionality of the hidden state")
	parser.add_option("--numLayers", action="store", type="int", dest="numLayers", default=3, help="Number of layers in the LSTM network")
	
	# Input Reader Params
	parser.add_option("--rootDir", action="store", type="string", dest="rootDir", default="../data/", help="Root directory containing the data")
	parser.add_option("-s", "--sequenceLength", action="store", type="int", dest="sequenceLength", default=5, help="Length of the sequence (this should split the input into equal chunks)")

	# Parse command line options
	(options, args) = parser.parse_args()
	print(options)

	if options.trainModel:
		print("Training model")
		train(options)

	if options.testModel:
		print("Testing model")
		test(options)
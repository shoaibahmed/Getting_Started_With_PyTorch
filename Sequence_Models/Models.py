import math
import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
	def __init__(self, sequenceLength, featureVectorLength, hiddenDims, numClasses, numLayers=3, dropout=0.0, bidirectional=False):
		super().__init__()
		self.numClasses = numClasses
		self.numLayers = int(numLayers * 2) if bidirectional else numLayers
		self.sequenceLength = sequenceLength
		self.featureVectorLength = featureVectorLength
		self.hiddenDims = hiddenDims
		self.bidirectional = bidirectional
		print ("Number of layers: %d" % self.numLayers)

		self.rnn = torch.nn.LSTM(featureVectorLength, hiddenDims, num_layers=numLayers, dropout=dropout, batch_first=True)
		self.linear = torch.nn.Linear((self.sequenceLength * self.hiddenDims * (2 if self.bidirectional else 1)), numClasses)
		
	def forward(self, x):
		# Split the input into sequences
		x = x.view(-1, self.sequenceLength, self.featureVectorLength)
		xShape = list(x.size())
		hidden = torch.zeros([self.numLayers, xShape[0], self.numClasses])
		outputs, hidden = self.rnn(x)
		# outputs = outputs[:, -1, :]
		outputs = outputs.contiguous().view(xShape[0], -1)
		outputs = self.linear(outputs)
		return outputs


class CNN(torch.nn.Module):
	def __init__(self, numClasses, inputShape, numFilters=(32, 64, 128)):
		super().__init__()
		assert len(inputShape) == 2, "Error: Input shape should be comprised of the sequence length followed by the number of channels!"
		self.inputShape = inputShape
		self.numClasses = numClasses
		self.numFilters = numFilters

		self.numLayers = len(numFilters)
		print("Number of convolutional layers: %d | Layer dimensions: %s" % (self.numLayers, str(self.numFilters)))

		self.convolutionalLayers = []
		self.poolingLayers = []

		outputSize = self.inputShape[0]
		for i in range(self.numLayers):
			if i == 0:
				conv = torch.nn.Conv1d(in_channels=self.inputShape[1], out_channels=self.numFilters[0], kernel_size=3, padding=1)
			else:
				conv = torch.nn.Conv1d(in_channels=self.numFilters[i-1], out_channels=self.numFilters[i], kernel_size=3, padding=1)

			self.poolingLayers.append(torch.nn.MaxPool1d(kernel_size=2, stride=2))

			self.convolutionalLayers.append(conv)
			outputSize = int(math.floor(outputSize / 2))  # Due to max-pooling

		outputSize = outputSize * numFilters[-1]  # For flattened output
		self.linear = torch.nn.Linear(outputSize, numClasses)


	def forward(self, x):
		# Apply the convolutional layers
		for i in range(self.numLayers):
			# Apply the convolution operation
			x = self.convolutionalLayers[i](x)

			# Apply the non-linearity
			x = torch.nn.ReLU(x)

			# Apply max-pooling
			x = self.poolingLayers[i](x)

		# Apply the linear layer
		outputs = self.linear(x)

		return outputs

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

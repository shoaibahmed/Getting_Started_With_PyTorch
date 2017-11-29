from optparse import OptionParser
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# PyTorch imports
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Command line options
parser = OptionParser()

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=28, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=28, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in the image")

parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=10000, help="Batch size")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning Rate")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=10, help="Number of classes")

parser.add_option("--numTrainingInstances", action="store", type="int", dest="numTrainingInstances", default=60000, help="Training instances")
parser.add_option("--numTestInstances", action="store", type="int", dest="numTestInstances", default=10000, help="Test instances")

# Parse command line options
(options, args) = parser.parse_args()
options.cuda = torch.cuda.is_available()
print (options)

kwargs = {'num_workers': 1, 'pin_memory': True} if options.cuda else {}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=options.batchSize, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=options.numTestInstances, shuffle=True, **kwargs)

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
		self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
		self.dropout = torch.nn.Dropout(p=0.35) # Probability of dropping the neuron
		# self.maxPool2D = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=0)

		# an affine operation: y = Wx + b
		self.fc1 = torch.nn.Linear(7 * 7 * 32, 256)
		self.dropoutTwo = torch.nn.Dropout(p=0.5) # Probability of dropping the neuron
		self.fc2 = torch.nn.Linear(256, 10)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		# x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
		x = self.dropout(x)
		# print ("Size of x: %s" % str(x.size())) # Should be 7x7x32
		x = x.view(-1, self.num_flat_features(x))
		x = F.tanh(self.fc1(x))
		x = self.dropoutTwo(x)
		x = self.fc2(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

def testModel():
	# Perform test step
	model.eval()
	lossTest = 0.0
	correct = 0
	for data, target in test_loader:
		if options.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = torch.autograd.Variable(data, volatile=True), torch.autograd.Variable(target)
		output = model(data)
		# lossTest += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
		lossTest += loss(output, target).data[0]
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	return lossTest

lossDict = {}
# Iterate over the different optimizers
# for optimizer in [("SGD", optimizerSGD), ("AdaDelta", optimizerAdadelta), ("Adam", optimizerAdam)]:
for optimizer in ["SGD", "AdaDelta", "Adam"]:
	print ("Selected optimizer: %s" % optimizer)
	lossDict[optimizer] = {"train" : [], "test": []}

	# Define the model
	model = Net()
	if options.cuda:
		model.cuda()

	# Define the loss
	loss = torch.nn.CrossEntropyLoss()

	# Define the optimizers
	if optimizer == "SGD":
		optim = torch.optim.SGD(model.parameters(), lr=1e-1)
	elif optimizer == "AdaDelta":
		optim = torch.optim.Adadelta(model.parameters(), lr=3e-1)
	else:
		optim = torch.optim.Adam(model.parameters(), lr=options.learningRate)

	for epoch in range(options.trainingEpochs):
		# Perform train step
		step = 0
		for batch_idx, (data, target) in enumerate(train_loader):
			# Compute test loss first since the error reported after optimization will be lower than the train error
			lossTest = testModel()

			# Perform the training step
			model.train()
			if options.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			optim.zero_grad()
			output = model(data)
			# loss = F.nll_loss(output, target)
			currentLoss = loss(output, target)
			lossTrain = currentLoss.data[0]
			currentLoss.backward()
			optim.step()

			print ("Epoch: %d | Step: %d | Train Loss: %f | Test Loss: %f" % (epoch, step, lossTrain, lossTest))
			lossDict[optimizer]["train"].append(lossTrain)
			lossDict[optimizer]["test"].append(lossTest)
			step += 1

# Plot the loss curves using Matplotlib
for optim in lossDict:
	mpl.style.use('seaborn')

	fig, ax = plt.subplots()
	ax.set_title('Optimizer: {!r}'.format(optim), color='C0')

	x = np.arange(0, len(lossDict[optim]["train"]))
	ax.plot(x, lossDict[optim]["train"], 'C0', label='Train', linewidth=2.0)
	ax.plot(x, lossDict[optim]["test"], 'C1', label='Test', linewidth=2.0)
	ax.legend()

	plt.savefig('./loss_curve_' + optim + ' .png', dpi=300)
	# plt.show()
	plt.close('all')
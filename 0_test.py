# PyTorch imports
import torch
import torch.nn.functional as F

# Define tensor
x = torch.autograd.Variable(torch.ones((5,5)), requires_grad=True)
print (x)

y = torch.autograd.Variable(torch.zeros(x.size()))
print (y)

for i in range(x.size()[0]):
	y[i, :] = x[i, :] * i

print (y)
out = torch.pow(y - x, 2).mean()
print ("Output: %s" % str(out))
out.backward()

print ("Custom gradient:")
customGrad = torch.autograd.Variable(torch.zeros(x.size()))
for i in range(x.size()[0]):
	# out = (2 * y - x).mean()
	customGrad[i, :] = 2 * (y[i, :] - x[i, :]) * i
print (customGrad / 25.0)

print ("Computed gradient:")
print (x.grad)

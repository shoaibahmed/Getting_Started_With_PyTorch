from functools import partial

import torch
from torch import nn
import torchvision.models as M
import pretrainedmodels


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class DenseNetFinetune(nn.Module):
	finetune = True

	def __init__(self, num_classes, net_cls=M.densenet121):
		super().__init__()
		self.net = net_cls(pretrained=True)

		# Add the deconvolutional layers
		self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

		self.score_fr = nn.Conv2d(4096, n_class, 1)
		self.score_pool3 = nn.Conv2d(256, n_class, 1)
		self.score_pool4 = nn.Conv2d(512, n_class, 1)

		self.upscore2 = nn.ConvTranspose2d(
			n_class, n_class, 4, stride=2, bias=False)
		self.upscore8 = nn.ConvTranspose2d(
			n_class, n_class, 16, stride=8, bias=False)
		self.upscore_pool4 = nn.ConvTranspose2d(
			n_class, n_class, 4, stride=2, bias=False)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.zero_()
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.ConvTranspose2d):
				assert m.kernel_size[0] == m.kernel_size[1]
				initial_weight = get_upsampling_weight(
					m.in_channels, m.out_channels, m.kernel_size[0])
				m.weight.data.copy_(initial_weight)

	def fresh_params(self):
		return self.net.classifier.parameters()

	def forward(self, x):
		return self.net(x)

densenet201_finetune = partial(DenseNetFinetune, net_cls=M.densenet201)
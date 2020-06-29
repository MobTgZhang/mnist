import gpytorch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import os
import numpy as np
import matplotlib.pyplot as plt
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm2d(32)
        self.fc3 = nn.Linear(32 * 7 * 7, 64)
        self.norm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.norm3(self.fc3(x)))
        return x
    
feature_extractor = LeNetFeatureExtractor().cuda()
# A gpytorch module is superclass of torch.nn.Module
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, n_features=64, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        # We add the feature-extracting network to the class
        self.feature_extractor = feature_extractor
        # The latent function is what transforms the features into the output
        self.latent_functions = LatentFunctions(n_features=n_features, grid_bounds=grid_bounds)
        # The grid bounds are the range we expect the features to fall into
        self.grid_bounds = grid_bounds
        # n_features in the dimension of the vector extracted (64)
        self.n_features = n_features
    
    def forward(self, x):
        # For the forward method of the Module, first feed the xdata through the
        # feature extraction network
        features = self.feature_extractor(x)
        # Scale to fit inside grid bounds
        features = gpytorch.utils.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # The result is hte output of the latent functions
        res = self.latent_functions(features.unsqueeze(-1))
        return res
    
# The AdditiveGridInducingVariationalGP trains multiple GPs on the features
# These are mixed together by the likelihoo function to generate the final
# classification output

# Grid bounds specify the allowed values of features
# grid_size is the number of subdivisions along each dimension
class LatentFunctions(gpytorch.models.AdditiveGridInducingVariationalGP):
    # n_features is the number of features from feature extractor
    # mixing params = False means the result of the GPs will simply be summed instead of mixed
    def __init__(self, n_features=64, grid_bounds=(-10., 10.), grid_size=128):
        super(LatentFunctions, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                              n_components=n_features, mixing_params=False, sum_output=False)
        #  We will use the very common universal approximator RBF Kernel
        cov_module = gpytorch.kernels.RBFKernel()
        # Initialize the lengthscale of the kernel
        cov_module.initialize(log_lengthscale=0)
        self.cov_module = cov_module
        self.grid_bounds = grid_bounds
        
    def forward(self, x):
        # Zero mean
        mean = Variable(x.data.new(len(x)).zero_())
        # Covariance using RBF kernel as described in __init__
        covar = self.cov_module(x)
        # Return as Gaussian
        return gpytorch.random_variables.GaussianRandomVariable(mean, covar)
    
# Intialize the model  
model = DKLModel(feature_extractor).cuda()
# Choose that likelihood function to use
# Here we use the softmax likelihood (e^z_i)/SUM_over_i(e^z_i)
# https://en.wikipedia.org/wiki/Softmax_function
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(n_features=model.n_features, n_classes=10).cuda()

# Simple DataLoader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)

# We use an adam optimizer over both the model and likelihood parameters
# https://arxiv.org/abs/1412.6980
optimizer = Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},  # SoftmaxLikelihood contains parameters
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, n_data=len(train_dataset))

def train(epoch):
    model.train()
    likelihood.train()
    
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.data[0]))
        train_loss += loss.data
    return train_loss/batch_size
def test():
    model.eval()
    likelihood.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = likelihood(model(data))
        pred = output.argmax()
        correct += pred.eq(target.view_as(pred)).data.cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.data.cpu().numpy() / len(test_loader.dataset)
n_epochs = 10
def save_config(loss_list,save_path,pic_save,acc_file):
    # save the plot picture
    x = np.linspace(0,len(loss_list),len(loss_list))
    plt.plot(x,loss_list)
    plt.savefig(os.path.join(save_path,pic_save))
    plt.show()
    with open(os.path.join(save_path,acc_file),"w") as fp:
        for k in range(len(loss_list)):
            fp.write(str(loss_list[k]) + "\n")
# While we have theoretically fast algorithms for toeplitz matrix-vector multiplication, the hardware of GPUs
# is so well-designed that naive multiplication on them beats the current implementation of our algorith (despite
# theoretically fast computation). Because of this, we set the use_toeplitz flag to false to minimize runtime
with gpytorch.settings.use_toeplitz(False):
    test_loss_list = []
    train_loss_list = []
    for epoch in range(1, n_epochs):
        out = train(epoch)
        train_loss_list.append(out)
        out = test()
        test_loss_list.append(out)
    save_path = "pytorch_save_dis"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pic_save = "loss.png"
    acc_file = "loss.txt"
    save_config(train_loss_list,save_path,pic_save,acc_file)
    pic_save = "accuracy.png"
    acc_file = "accuracy.txt"
    save_config(test_loss_list,save_path,pic_save,acc_file)

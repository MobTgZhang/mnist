import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(320, 10)
    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return F.log_softmax(x,dim = 1)

model = Net()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    loss_all = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_all += loss.data
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    loss_all = loss_all / len(test_loader.dataset)
    return loss_all.cpu().data.numpy()
def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.data.numpy() / len(test_loader.dataset)
def save_config(loss_list,save_path,pic_save,acc_file):
    # save the plot picture
    x = np.linspace(0,len(loss_list),len(loss_list))
    plt.plot(x,loss_list)
    plt.savefig(os.path.join(save_path,pic_save))
    plt.show()
    with open(os.path.join(save_path,acc_file),"w") as fp:
        for k in range(len(loss_list)):
            fp.write(str(loss_list[k]) + "\n")
def main():
    test_loss_list = []
    train_loss_list = []
    for epoch in range(1, 20):
        out = train(epoch)
        train_loss_list.append(out)
        out = test()
        test_loss_list.append(out)
    save_path = "torch_save"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pic_save = "loss.png"
    acc_file = "loss.txt"
    save_config(train_loss_list,save_path,pic_save,acc_file)
    pic_save = "accuracy.png"
    acc_file = "accuracy.txt"
    save_config(test_loss_list,save_path,pic_save,acc_file)
if __name__ == '__main__':
    main()
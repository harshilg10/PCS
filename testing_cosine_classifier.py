import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn

from torch import relu

from torch.utils.data import DataLoader, Dataset

import torchvision.datasets as dsets
import torchvision.transforms as transforms

# GPU settings
device = torch.device(0)
device

# K-shot N-ways
K = 10
N = 1
left_class = 7

data_transform = transforms.Compose([transforms.ToTensor()])

# import the `MNIST datasets`
mnist_train = dsets.MNIST(root='data',
                          train=True,
                          transform=data_transform,
                          download=True)

mnist_test = dsets.MNIST(root='data',
                          train=False,
                          transform=data_transform,
                          download=True)

# build the `DataLoader`
train_data_loader = DataLoader(mnist_train, batch_size=2**9, num_workers=14)
test_data_loader = DataLoader(mnist_test, batch_size=mnist_test.data.shape[0], num_workers=14)

# Label Encoder
label_encoder  = preprocessing.LabelEncoder()

targets = list(range(0, 10, 1))
targets.pop(left_class)
targets = np.array(targets).reshape(-1, 1)

label_encoder.fit(targets)
#####
print("label encoder is fitted")
class Model(nn.Module):
    def __init__(self, in_size=28, embedding_feature_size=2, n_classes=10):
        super().__init__()
        
        # Data properties
        in_channels = 1
        
        # Define layers
        # 1
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1)
        self.bn1 = nn.Dropout(0.2) #nn.BatchNorm2d(12)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.Dropout(0.2) #nn.BatchNorm2d(24)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.Dropout(0.0) #nn.BatchNorm2d(48)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4
        self.last_embedding_layer = nn.Linear(48*3*3, embedding_feature_size)
        
        # Each `w` is a representation vector of each class.
        self.Wstar_layer = nn.Linear(embedding_feature_size, n_classes, bias=False)  # 'bias=False' allows the weights to be the pure representation vectors (not affected by the bias)
        
    def forward(self, x):
        #============================
        # Feature Extractor
        h1 = self.maxpool1(relu(self.bn1(self.conv1(x))))
        h2 = self.maxpool2(relu(self.bn2(self.conv2(h1))))
        h3 = self.maxpool3(relu(self.bn3(self.conv3(h2))))
        
        batch_size = x.shape[0]
        h3 = h3.view(batch_size, -1)  # 1
        
        self.z = self.last_embedding_layer(h3)
        norm_z = self.z / torch.norm(self.z, p=2, dim=1, keepdim=True)#.detach() # 2)
        
        # 1) flattne layer
        # 2) representation vector of `x`; Note `.detach()`: ; We don't want the l2-norm function to be involved with the gradient descent update.
        
        #============================
        # Cosine-similarity Classifier
        Wstar = self.Wstar_layer.weight.T#.detach()  # Note `.detach()`
        norm_Wstar = Wstar / torch.norm(Wstar, p=2, dim=0, keepdim=True)
        
        cosine_similarities = torch.mm(norm_z, norm_Wstar)
        # Note that `CrossEntropyLoss()` = `LogSoftmax` + `NLLLoss`
        return cosine_similarities

model = Model(embedding_feature_size=64, n_classes=9)
print("Model is initialized!")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

decayRate = 1.0
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                         gamma=decayRate)

# temperature parameter
tau = 0.1
print("Just before loop")
for x, y in train_data_loader:
    #print(x); print(y); print(x.max()); print(x.shape)
    break

from scipy.special import softmax

yhat = model(x)
print("...")
yhat = yhat[0, :].to("cpu").detach().numpy()
x_ = np.arange(yhat.size)

### TRAIN
# settings
n_epochs = 6
def leave_out_a_class(x, y, left_class):
    "leave out some class for the few-shot learning"
    indices = (y != left_class)
    return x[indices, :, :, :], y[indices]

train_hist = {"epochs": [], "loss_per_epoch": [], "loss": [], "test_acc": []}
print("Just before training!")
for epoch in range(n_epochs):
    model.train()
    
    loss_per_epoch = 0
    iters = 0
    for x, y in train_data_loader:
        x, y = leave_out_a_class(x, y, left_class)
        y = torch.tensor(label_encoder.transform(y))
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        yhat = model(x)
        
        loss = criterion(yhat/tau, y)
        loss.backward()
        
        optimizer.step()
        
        # data storage
        loss_per_epoch += loss
        iters += 1
        train_hist["loss"].append(loss)
        #print(round(loss.item(), 2), end=" ")
            
    train_hist["epochs"].append(epoch)
    train_hist["loss_per_epoch"].append(loss_per_epoch)
    
    # validation
    with torch.no_grad():
        model.eval()
        test_acc = 0
        for x_test, y_test in test_data_loader:
            x_test, y_test = leave_out_a_class(x_test, y_test, left_class)
            y_test = torch.tensor(label_encoder.transform(y_test))
            x_test, y_test = x_test.to(device), y_test.to(device)
            yhat = torch.argmax(model(x_test.to(device)), axis=1)
            test_acc += np.mean((yhat.to("cpu") == y_test.to("cpu")).numpy())
        train_hist["test_acc"].append(test_acc)
    
    my_lr_scheduler.step()
    
    print("epoch: {}, loss: {:0.3f}, test_acc: {:0.3f} | lr: {:0.4f}".format(epoch, loss_per_epoch/iters, test_acc, optimizer.param_groups[0]['lr']))


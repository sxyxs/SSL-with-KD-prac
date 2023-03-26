import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 10
num_workers = 4   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-4
epochs = 2

image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),  
     # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])

class MDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("./dataset/19kul_train.csv")
test_df = pd.read_csv("./dataset/mnist_test.csv")

train_data = MDataset(train_df, data_transform)
test_data = MDataset(test_df, data_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x
def softmax(x):
    s = torch.exp(x)
    return s / torch.sum(s, dim=1, keepdim=True)

model = torch.load("11ep_1kdata_sup_Model.pkl")
model = model.cuda()
print("a")
criterion = nn.CrossEntropyLoss()
model.eval()
print("a")

val_loss = 0
gt_labels = []
pred_labels = []
with torch.no_grad():
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        output = model(data)
        # preds = torch.argmax(output, 1)
        preds = softmax(output)
        #print(np.sum(preds.cpu().data.numpy()))
        # print(len(preds))
        pred_labels.append(preds.cpu().data.numpy())

pred_labels =np.concatenate(pred_labels)
print(len(pred_labels))

df1 = pd.DataFrame(data=pred_labels,
                      columns=['0','1','2','3','4','5','6','7','8','9'])
df1.to_csv('dataset/pseudo_label.csv',index=False)

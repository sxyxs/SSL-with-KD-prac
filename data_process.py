# 用mmengine2.0 
# huggingface transformer Timmzuo wei backbone 

import torchvision
import torch
from torchvision import datasets, transforms

data_train = datasets.MNIST(root = "./data/",
                            transform=transforms.ToTensor(),
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                           train = False)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train.__getitem__(0),
                                                batch_size = 64,
                                                shuffle = True,
                                                 num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True,
                                                num_workers=2)

print(len(data_train[0][0][0][0]))

# 1.想办法只留下一部分数据 把label抓出来


print(data_train.__getitem__(0)[1])
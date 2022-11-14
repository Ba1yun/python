import torch
from torch import nn
from Net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets,transforms
import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
#数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.ToTensor()
])
#加载训练数据集   pytorch已经有了需要加载进来
train_dataset=MNIST(root='./data',train=True,transform=data_transform,download=True) #训练集下载下来了
train_dataloader=DataLoader(dataset=train_dataset,batch_size=16,shuffle=True, num_workers=0)#训练数据加载器
#加载测试数据集
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
#print(len(test_dataset))
test_dataloader=DataLoader(dataset=test_dataset,batch_size=16,shuffle=True, num_workers=0)#测试数据加载器

#如果有显卡就转到GPU
device="cuda" if torch.cuda.is_available() else "cpu"

#调用搭建好的网络模型，将模型数据转到GPU上面
model=MyLeNet5().to(device)
#定义一个损失函数--交叉熵损失
loss_fun=nn.CrossEntropyLoss()
#定义一个优化器
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)  #le是学习率
#学习率，每隔10轮，变为原来的0.1
lr_scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
#训练函数
def train(dataloader,modle,loss_fun,optimizer):   #传入上述的数据集，模型，损失函数，优化器
    loss,current,n=0,0,0
    for batch,(X,y) in enumerate(dataloader):   #X:就是比如10张图片，y是标签，取出来送入神经网络
        #前向传播
        X,y=X.to(device),y.to(device)
        output=model(X)
        cur_loss=loss_fun(output,y)  #和真实标签做一个交叉熵的计算
        _,pred=torch.max(output,axis=1) #通过torch.max这个函数找到最大的概率值然后索引输出是第几章图片
        cur_acc=torch.sum(y==pred).numpy()/len(train_dataloader)  #如果y==pred就是预测准确的话返回1，除以批次所有的图片（16张）计算这个批次的精确度
        #反向传播
        optimizer.zero_grad()#  梯度清零
        cur_loss.backward()
        optimizer.step()
        loss+=cur_loss.item()#将这一批次的所有损失值加在一起
        current+=cur_acc.item()#将该批次的精度值加在一起
        n=n+1  #批次次数
    print("train_loss"+str(loss/n))
    print("train_acc" + str(current / n))  #求整个精确度，应该是是不是应该是accurate？平写错了？

#测试
def val(dataloader,model,loss_fun):
    model.eval()
    loss,current,n=0,0,0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):  # X:就是比如10张图片，y是标签，取出来送入神经网络
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            # cur_loss = loss_fun(output, y)  # 和真实标签做一个交叉熵的计算
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / len(train_dataloader)
            # optimizer.zero_grad()  # 梯度清零
            # cur_loss.backward()
            # optimizer.step()
            # loss += cur_loss.item()
            # current += cur_acc.item()
            n = n + 1
        # print("val_loss" + str(loss / n))
        print("val_acc" + str(cur_acc / n))
        return cur_acc/n  #返回一个精确度，最终评判模型好坏是看测试集的精确度

#开始训练
epoch=50   #训练轮次
min_acc=0
for i in range(epoch):
    print(f'epoch{i+1}\n----------------')
    train(train_dataloader,model,loss_fun,optimizer)
    a=val(test_dataloader,model,loss_fun)
    #保存最好的模型权重
    if a>min_acc:

        # folder='seve_model'   #保存最好模型的文件
        # if not os.path.exists(folder):  #不存在就创建文件
        #     os.mkdir('save_model')
        min_acc=a
        # print('save best model')
    torch.save(model.state_dict(),'./save_model.pth')   #保存
    print('Done')


'''
1.nn是卷积神经网络的基网络框架里面包括了卷积操作激活函数等一系列操作
2.lr_scheduler：学习率是指对前向传播得到的误差的利用率，因为反向传播也就是梯度下降是根据误差来开始的误差越大收敛速度越快但不稳定
所以一般这个学习率是随着迭代次数的增加而减少，一开始学习率大一点然后越到后面越稳定收敛效果最好，常见学习率轮数减缓、指数减缓、分数减缓
3.torchvision.datasets,transforms，torchvision独立于torch，需要单独安装，主要提供三个内容（1.models如典型的AlexNet、VGG、ResNet）
（2.数据集加载，继承与torch.utils.data.Dataset如：MNIST、coco、imagenet）,(3.transforms：数据预处理包括Tensor，pil,image)
4.batch_size:就是一次读取图片的多少，批量读取大小。
5.shuffle:是否打乱，True/false
6.num_workers=0,多少个子线程用于加载数据,默认是0,表示只在主线程加载数据
7.transforms.Compose，就是把图像预处理步骤合在一起
8.import as的引用方法
9.transforms.ToTensor()：预处理将图片转化成Tensor格式，并归一到0-1之间
10.root=./data表示保存到当前目录下的data文件夹
11.step_size：（学习率减缓的一种）代表每隔多少迭代次数进行一次lr*gamma,若step_size=10，gamma=0.1代表迭代10此次学习率就降低原来的0.1倍
'''
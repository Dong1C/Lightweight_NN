import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt 
import copy
import models
from data_loader import load_data

# set the models
model_classes = {
    "squeezenet": models.squeezenet_model,
    "mobilenet": models.mobilenet_model,
    "xceptionnet": models.xceptionnet_model,
    "shufflenet": models.shufflenet_model
}

# set the dataloaders, dataset_sizes, class_names
dataloaders, dataset_sizes, class_names = load_data()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train
for mdn, model in model_classes.items():
    print(f"--------------------------{mdn} train begin--------------------------")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 开始训练模型
    num_epochs = 5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 初始化记录器
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    
        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            running_loss = 0.0
            running_corrects = 0
    
            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                # 零参数梯度
                optimizer.zero_grad()
    
                # 前向
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    # 只在训练模式下进行反向和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()
    
            # 记录每个epoch的loss和accuracy
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
    
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    torch.save(best_model_wts, f'./models_pms/{mdn}_bw')
    epoch = range(1, len(train_loss_history)+1)
 
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(epoch, train_loss_history, label='Train loss')
    ax[0].plot(epoch, val_loss_history, label='Validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Loss')  # 添加标题
    
    ax[1].plot(epoch, train_acc_history, label='Train acc')
    ax[1].plot(epoch, val_acc_history, label='Validation acc')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Accuracy')  # 添加标题
    
    plt.savefig(f"./log_dir/loss-acc_{mdn}.jpg", dpi=300,format="jpg")
    
    print('Best val Acc: {:4f}'.format(best_acc))
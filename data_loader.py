import torch
from torchvision import datasets, transforms
import os

def load_data(data_dir='./data/MTB', batch_size=32, img_size=(100, 100)):
    img_height = img_size[0]
    img_width = img_size[1]
    
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_height),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 加载数据集
    full_dataset = datasets.ImageFolder(data_dir)
    
    # 获取数据集的大小
    full_size = len(full_dataset)
    train_size = int(0.7 * full_size)  # 假设训练集占80%
    val_size = full_size - train_size  # 验证集的大小
    
    # 随机分割数据集
    torch.manual_seed(0)  # 设置随机种子以确保结果可重复
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 将数据增强应用到训练集
    train_dataset.dataset.transform = data_transforms['train']
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes
    
    return dataloaders, dataset_sizes, class_names


if __name__ == '__main__':
    a, b, c = load_data()
    print(a, b, c)
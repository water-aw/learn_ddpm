import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        # 将图像缩放到统一尺寸，确保图像具有相同大小，以适配神经网络输入要求
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # 对图像进行随机水平翻转（50%概率），用于数据增强，提升模型泛化能力
        transforms.RandomHorizontalFlip(),
        # 将图像转换为PyTorch张量（Tensor），并将像素值从[0, 255]缩放到[0, 1]范围
        transforms.ToTensor(), # Scales data into [0,1]
        # 自定义变换，将张量的数值范围从[0, 1]映射到[-1, 1]
        transforms.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1]
    ]
    # 组合变换操作
    data_transform = transforms.Compose(data_transforms)
    # 加载训练集和测试集
    train = torchvision.datasets.StanfordCars(root='.', split='train', download=False, transform=data_transform)
    test = torchvision.datasets.StanfordCars(root='.', split='test', download=False, transform=data_transform)
    # 合并数据集
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        # 归一化还原
        transforms.Lambda(lambda t: (t + 1) / 2),
        # 转换维度
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        # 放大数值
        transforms.Lambda(lambda t: t * 255.),
        # 将张量类型转换为数组类型
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        # 转为PIL图像
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)





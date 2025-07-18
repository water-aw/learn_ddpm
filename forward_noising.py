import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplot
from dataloader import load_transformed_dataset, show_tensor_image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def linera_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    定义了一个线性增长的Beta调度
    :param timesteps: 时间步数，表示整个过程分为多少个阶段
    :param start: 起始Beta值，默认为0.0001
    :param end: 结束Beta值，默认为0.02
    :return: 在[start, end]之间均匀生成timesteps个值，实现Beta实践线性递增
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Gather specific values from the tensor vals based on the indices
    in t, and reshape them to match the shape of an image batch
    for broadcasting operations.
    :param vals: A 1D tensor with length equal to the total number of time steps
    :param t: A tensor containing indices for which we want ot retrieve
    :param x_shape: The shape of the input image tensor([batch_size, channels, height, width])
    :return:
    """
    # Get batch size(time step)
    batch_size = t.shape[0]
    # Get values based on indices
    out = vals.gather(-1, t.cpu())
    # Reshape for broadcasting
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    return the noisy version of it
    x_0: 初试图像张量，形状通常为[batch_size, channels, height, width]
    t: 时间步索引，表示当前扩散过程的阶段，通常是一个整数或张量
    """
    # 生成噪声
    noise = torch.randn_like(x_0)
    # 获取时间步参数
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    # 构造带噪图像
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define beta schedule
# 定义时间步总数为300
T = 300
# 生成一个包含300个元素的beta值列表
betas = linera_beta_schedule(timesteps=T)
print(type(betas))

# Pre-calculate different terms for closed form
    # 表示每一步保留原始数据信息的比例
alphas = 1. - betas
# 计算了张量alphas在第0轴上的累乘结果
    # 在第0轴上的累乘，表示向前扩散过程中累积的保留率
alphas_cumprod = torch.cumprod(alphas, axis=0)
    # 将alphas_cumprod向右平移一位，用于后续计算扩散的后验方差
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    # 计算alphas倒数的平方根，用于去噪过程中的缩放
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    # 计算累乘alphas_cumprod的平方根，用于构建加噪公式
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    # 计算1 - alphas_cumprod的平方根，用于构建噪声项
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    # 计算扩散模型中每一步的后验方差，用于反向扩散过程的方差控制
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Simulate forward diffusion
    # 加载数据集并取出一张图像
dataloader = load_transformed_dataset()
    # 从数据集中取出第一个批次的第一张图像用于模拟向前扩散过程
image = next(iter(dataloader))[0]

# 创基一个大小为15x15英尺的图像窗口
plt.figure(figsize=(15,15))
# 关闭坐标轴显示
plt.axis('off')
# 定义要显示的图像数量为10张
num_images = 10
# 计算每隔stepsize步显示一张图像
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    # 将idx转换为整数类型的张量，表示当前时间步
    t = torch.Tensor([idx]).type(torch.int64)
    # 设置绘图区域，按列排列多个子图
        # 第一个参数表示创创建一行网格
        # 第二个参数表示子图的网格数量为图像+1，通常用于显示原图或说明
        # 第三个图像控制图像的额显示间隔
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    # 模拟向前扩散过程，并获得加噪后的图像和噪声。
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)

plt.show()
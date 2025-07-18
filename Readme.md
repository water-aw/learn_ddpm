本项目为复刻diffusion model模型的项目，非原创项目。
参考项目地址：https://github.com/chunyu-li/ddpm.git

- learn_ddpm
  - dataset
    导入斯坦福汽车的数据集，但是该数据集的官网已经失效，需要手动导入官网。
    这里提供百度网盘版本。请务必按我给出的格式引入图片文件，否则会出现找不到数据集的错误。
    通过网盘分享的文件：stanford_cars.7z
    链接: https://pan.baidu.com/s/1GBfJkKWrcW7Pkjxr1EVj7Q?pwd=nhzh 提取码: nhzh
  - dataloader
    加载数据集
  - forward_noising
    模拟扩散模型的向前过程
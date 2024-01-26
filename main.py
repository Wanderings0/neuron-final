from neuron import h, gui
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from simulate import simulate_neuron_model



# for sec in h.allsec():  # 遍历所有的段
#     print('Section name: {}, Section type: {}'.format(sec.name(), sec.hname()))
#     for seg in sec:     # 遍历段中的点位
#         print('  at position:', seg.x, 'has diameter:', seg.diam)


# 根据pixel的值来生成一个IClamp对象并返回


# 设定随机种子以确保可重复性
torch.manual_seed(0)

# 设置Fashion MNIST数据集的加载
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 32 # 你可以根据你的系统配置调整批量大小

# 加载Fashion MNIST数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 获取所有标签以进行分层采样
targets = trainset.targets

# 使用StratifiedShuffleSplit来获得1%的数据并保持类别平衡
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.01, random_state=0)
for train_index, test_index in sss.split(targets, targets):
    subset_indices = test_index

# 创建一个数据子集
train_subset = Subset(trainset, subset_indices)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

targets2 = testset.targets

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.01, random_state=0)
for train_index, test_index in sss2.split(targets2, targets2):
    subset_indices2 = test_index

test_subset = Subset(testset, subset_indices2)


def compute_gradient(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    grad_x = ndimage.convolve(image, sobel_x)
    grad_y = ndimage.convolve(image, sobel_y)
    grad = np.uint8(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return grad

class GradientDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.squeeze().numpy()*255  # 转换为numpy数组
        image = np.uint8(image.sum(axis=0)/3)
        # gradient_magnitude = compute_gradient(image)
        # gradient_magnitude = torch.from_numpy(gradient_magnitude).float().unsqueeze(0)  # 转回torch张量
        # return gradient_magnitude, label
        return image, label

    def __len__(self):
        return len(self.dataset)

train_grad_set = GradientDataset(train_subset)
test_grad_set = GradientDataset(test_subset)
# 使用数据加载器加载数据 gradient_magnitude
trainloader = DataLoader(train_grad_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_grad_set, batch_size=batch_size, shuffle=False)
# no grad
# trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# 存储模拟输出和标签
simulated_outputs = []
labels = []
# # 遍历数据集
for images, lbls in trainloader:
    # print(f'lbls.shape is {lbls.shape}')
    for index, image in enumerate(images):
        # 转化成灰度图像
        print(f'image.shape is {image.shape}')
        image = image.squeeze().numpy()
        # image = np.uint8(image.sum(axis=0)/3)
        print(image.shape)

        # gradient_magnitude = torch.from_numpy(gradient_magnitude).float().unsqueeze(0)
        output = simulate_neuron_model(image)
        simulated_outputs.append(output)
        labels.append(lbls[index])



# 转换为Tensor
# simulated_output 是一个列表，每个元素的形状都是（1201，）的numpy数组
# labels 是一个列表，每个元素是一个形状为[3]的张量
# 我们如何将它们保存以便供未来的神经网络作为数据集使用？
# 打包simulated_output和labels到一个.npz文件
np.savez('trainset_loc_no_grad.npz', simulated_outputs=simulated_outputs, labels=labels)

test_outputs = []
test_labels = []

for images, lbls in testloader:
    for index, image in enumerate(images):
        image = image.squeeze().numpy()
        # image = np.uint8(image.sum(axis=0)/3)
        output = simulate_neuron_model(image)
        test_outputs.append(output)
        test_labels.append(lbls[index])

np.savez('testset_loc_no_grad.npz', simulated_outputs=test_outputs, labels=test_labels)


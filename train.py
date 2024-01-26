import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import wandb
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.init as init

class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        return self.layer(x)


class FCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FCNN, self).__init__()
        self.layer1 = nn.Linear(in_channels, hidden_channels)
        self.layer2 = nn.Linear(hidden_channels, out_channels)
        # 初始化第一层的权重和偏置
        # init.kaiming_normal_(self.layer1.weight, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.layer1.bias, 0)
        # # 初始化第二层的权重和偏置
        # init.kaiming_normal_(self.layer2.weight, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.layer2.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1) # 展平输入
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    

def svmloss(scores: torch.Tensor, label: torch.Tensor):
    '''
    compute SVM loss
    input:
        scores: output of model 
        label: true label of data
    return:
        svm loss
    '''
    # compute the svm loss from scratch
    # print(label.shape)
    batch_size = scores.size(0)

    # Gather the correct scores for each input
    correct_scores = scores[torch.arange(batch_size), label].unsqueeze(1)  # shape: [batch_size, 1]
    # Calculate the margins for all classes in one vector operation
    margins = torch.maximum(torch.tensor(0), scores - correct_scores + 1)  # delta = 1
    margins[torch.arange(batch_size), label] = 0  # Do not consider correct class in loss

    # Sum over all incorrect classes
    loss = torch.sum(margins) / batch_size

    return loss

def crossentropyloss(logits: torch.Tensor, label: torch.Tensor):
    '''
    Object: implement Cross Entropy loss function
    input:
        logits: output of model, (unnormalized log-probabilities). shape: [batch_size, c]
        label: true label of data. shape: [batch_size]
    return: 
        cross entropy loss
    '''
    # softmax function
    exp_logits = torch.exp(logits - torch.max(logits,dim=1,keepdim=True).values)
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    # cross entropy

    batch_size = logits.size(0)
    log_probs = -torch.log(probs[torch.arange(batch_size), label])

    loss = torch.sum(log_probs) / batch_size

    return loss


def evaluate(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / total

def train(model, loss_function, optimizer, scheduler, args):
    wandb.init(project="neuron-final", config=args)
    # wandb.init(project='SNN MNIST',config=args)
    wandb.run.name = f'neuron {args.path} {args.seed}'

    device = torch.device(args.device)
    model.to(device)

    # 从npz文件中加载数据并转换成dataloader
    # 保存的时候使用的语句和路径是np.savez('trainset_loc_no_grad.npz', simulated_outputs=simulated_outputs, labels=labels)
    # 加载训练和测试数据集
    if args.path == 'mod':
        train_data = np.load('trainset_mod_no_grad.npz')
        test_data = np.load('testset_mod_no_grad.npz')
    elif args.path == 'loc':
        train_data = np.load('trainset_loc_no_grad.npz')
        test_data = np.load('testset_loc_no_grad.npz')

    # 将numpy数组转换成torch张量
    train_inputs = torch.from_numpy(train_data['simulated_outputs']).float()
    train_labels = torch.from_numpy(train_data['labels']).long()
    test_inputs = torch.from_numpy(test_data['simulated_outputs']).float()
    test_labels = torch.from_numpy(test_data['labels']).long()

    # 创建TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    # 创建DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_accuracy = 0.0
    best_model_state = None

    for epoch in range(args.epoch):
        model.train()
        # running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()

        avg_loss, train_accuracy = evaluate(model, train_loader, loss_function, device)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
        print(f'Epoch {epoch+1} - Loss: {avg_loss:.6f}, Training Accuracy: {train_accuracy * 100:.2f}%')
        print(f'Epoch {epoch+1} - Loss: {test_loss:.6f}, Testing Accuracy: {test_accuracy * 100:.2f}%')
        
        # Adjust the learning rate
        scheduler.step()

        # Log the loss and accuracy for the current epoch to wandb
        wandb.log({"train_loss":avg_loss,"train_acc": train_accuracy, "test_acc":test_accuracy,"epoch": epoch + 1})

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            # best_model_state = model.state_dict()

    if best_model_state is not None:
        # torch.save(best_model_state, 'best_model.pth')
        # wandb.save('best_model.pth')
        print('Saved best model with accuracy: {:.2f}%'.format(best_accuracy * 100))
    
    print('Finished Training')
    wandb.finish()


def test(model, loss_function, args):
    device = torch.device(args.device)
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    test_loss, test_accuracy = evaluate(model, testloader, loss_function, device)
    print(f'Accuracy of the network on the 10000 test images: {test_accuracy * 100:.2f}%')

def main():

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='fcnn')
    parser.add_argument('--loss', type=str, default='crossentropyloss')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--path', type=str, default='loc')
    parser.add_argument('--seed', type=int, default=0)



                        
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    #为所有环节设置随机数种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("-"*10)
    print(f'{args.model} {args.loss} {args.optimizer} {args.scheduler} begin!')
    print("-"*10)

    # create model
    if args.model == 'linear':
        model = LinearClassifier(701, 10)
    elif args.model == 'fcnn':
        model = FCNN(701, 128, 10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, eval(args.loss), optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, eval(args.loss), args)
    else: 
        raise AssertionError
    
    print("-"*10)
    print(f'{args.model} {args.loss} {args.optimizer} {args.scheduler} finish!')
    print("-"*10)
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --loss=crossentropyloss --optimizer=adamw --scheduler=step

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from utils.model import MnistModel
from utils.dataset import MnistData

import argparse



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()
    return args


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # if args.dry_run:
            #     break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    args = parseArgs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda: device = torch.device("cuda")
    else: device = torch.device("cpu")

    train_kwargs = {'batch_size':args.batch_size}
    test_kwargs = {'batch_size':args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MnistData(root='./data/', train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_dataset = MnistData(root='./data/', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = MnistModel().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "weights/mnist_cnn.pth")

    # Step 2: Just-in-time compilation
    model.to(torch.device("cpu"))
    # model.eval()
    input_shape = [1,1,28,28]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    scripted_model.save('weights/mnist_cnn_scripted.pth')

if __name__ == '__main__':
    main()

import torch
import torch.optim as optim
import torch.nn.functional as F

from model.utils import get_data_loaders


def train(model, device, train_loader, optimizer, epoch, loss_func, subnet=None):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if subnet is None:
            model.sample_subnet()
        else:
            model.sample_subnet(*subnet)

        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_func, subnet):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss_func(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Testing Subnet {}, Accuracy: {}/{} ({:.0f}%)'.format(
        subnet, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def train_test(supernet, epochs=20, subnet=None):

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders()

    test_acc = []

    if subnet is None:
        model = supernet.to(device)
    else:
        supernet.sample_subnet(*subnet)
        model = supernet.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_func = F.cross_entropy

    if subnet is None:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, loss_func)

            test_acc_ = []
            for subnet in [(x, y) for x in range(3) for y in range(3)]:
                model.sample_subnet(*subnet)
                test_acc_.append(test(model, device, test_loader, loss_func, subnet))
            test_acc.append(test_acc_)

    else:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, loss_func, subnet=subnet)
            test_acc.append(test(model, device, test_loader, loss_func, subnet))

    return test_acc

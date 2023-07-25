import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader


def plot_accuracies(mode, accuracies, title, save_path):
    if mode == 'oneshot':
        accuracies = list(map(list, zip(*accuracies)))

    fig, ax = plt.subplots()
    for idx, acc in enumerate(accuracies):
        ax.plot(acc, '-s', label=f'Subnet {idx+1}')
    ax.grid()
    ax.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    fig.savefig(save_path)
    plt.show()


def generate_accuracy_table(oneshot_acc, standalone_acc, output_path, epochs=20):
    index = [f"Epoch {epoch + 1}" for epoch in range(epochs)]
    data = {
        "subnet": index,
        "oneshot": [],
        "standalone": []
    }
    oneshot_acc = list(map(list, zip(*oneshot_acc)))

    for i in range(9):
        data["oneshot"].append(max(oneshot_acc[i]))
        data["standalone"].append(max(standalone_acc[i]))

    df = pd.DataFrame(data, index=index)

    styled_table = df.style.background_gradient(cmap='Blues')
    image = styled_table.to_image()
    image.save(output_path)

    return df

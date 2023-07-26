# One-Shot Neural Architecture Search

This repository contains code for training a SuperNet using one-shot neural architecture search (NAS) on the MNIST dataset. One-shot NAS is a technique for automatically discovering high-performing neural network architectures by training a single "SuperNet" that contains all possible sub-networks, and then evaluating the performance of each sub-network on a validation set. This approach can significantly reduce the computational cost of architecture search, as it requires training only one network instead of many. In this project, we use a SuperNet that contains 9 different architectures, and compare the accuracy and training time of sub-networks trained using the SuperNet to those trained independently.

## Task

The task at hand is to train a SuperNet on the MNIST dataset using 9 different architectures and compare its performance with independently trained networks.

## SuperNet Structure

The SuperNet is a set of different layers, as shown in the figure below. The architectures that can be obtained from the SuperNet are shown on the right-hand side of the figure when changing the first block.

![SuperNet architecture](pics/supernet.png "SuperNet architecture")

The convolutions located in the variable-length block (highlighted in yellow) do not change the height and width of the input tensor, as well as the number of channels.

For variable-length blocks, the sequence of convolutions is fixed, there are no gaps or changes in the order of convolutions in the block (Allowed: 1, 1-> 2, 1-> 2-> 3; NOT allowed: 2, 3, 2-> 3, 1-> 3, 2-> 1, 3-> 2-> 1, etc.)

The figure below shows the architecture of the variable block of the SuperNet, as well as its specific implementations.

![NAS Block architecture](pics/NAS_block.png "NAS block architecture")

## Script Structure

The script that runs the training and testing is located in the `main.py` file. It calls the `train_test` function from the `train_test` module, passing an instance of the `SuperNet` class as an argument. After training and testing the SuperNet, the script calls the `train_test` function for each of the 9 possible subnets, passing an instance of `SuperNet` and the corresponding subnet index as arguments.

# How to Run the Script
To add this repository to your local machine, you can follow these steps:

**Step 1.** Clone the repository by running the following command in the terminal:
`git clone https://github.com/nik1everyday/OneShotNeuralArchitectureSearch.git`

**Step 2.** Navigate to the repository directory:
`cd OneShotNeuralArchitectureSearch`

**Step 3.** Install the required dependencies using Poetry:
`poetry install`

This will install all the required dependencies specified in the pyproject.toml file.

**Step 4.** Run the script by calling the following command in the terminal:
`python main.py`
This will train and test the SuperNet and each of the 9 subnets on the MNIST dataset, and generate the accuracy results and comparison table.

Note that if you add or remove dependencies from the pyproject.toml file, you will need to re-run poetry install to update the virtual environment with the new dependencies.


# Results

The SuperNet and each of the 9 subnets were trained and tested on the MNIST dataset. The accuracy results for each subnet obtained during the experiments are presented below.

## One Shot Subnets

![One Shot Training Progress](pics/one-shot-subnets.png)

## Standalone Subnets

![Standalone Training Progress](pics/standalone-subnets.png)

## Comparison Table

![One Shot vs Standalone Subnets Top-1 Accuracies Table](pics/oneshot_vs_standalone_table.png)

From the experiment results, it can be seen that the subnets trained using the SuperNet achieve comparable accuracy to independently trained subnets but train significantly faster. However, the accuracy of the subnets trained using the SuperNet may be slightly lower than the accuracy of independently trained subnets.

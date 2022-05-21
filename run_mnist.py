#%% Preprocess and download datasets

# Variables
DATASET_PATH = './data/'
RESULT_PATH = './result/'

EPOCHS = 15
BATCH_SIZE = 16

# Import the necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm

# Loading the MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Resize((224, 224)),
                                transforms.Normalize((0.5), (0.5))
                               ])

# Download and load the training data
trainset = datasets.MNIST(DATASET_PATH, download = True, train = True, transform = transform)
testset = datasets.MNIST(DATASET_PATH, download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True)

#%% Define the model and the optimizer
from alexnet import AlexNet
from lenet import LeNet
from optim import SGD

model = LeNet(input_channel=1, output_class=10)
optimizer = SGD(model, lr=1e-5, momentum=0.9)

#%% Training and evaluating process

# Initiate the timer to instrument the performance
timer_start = time.process_time_ns()
epoch_times = [timer_start]

train_losses, test_losses, accuracies = [], [], []

for e in range(EPOCHS):
    running_loss = 0
    print("Epoch: {:03d}/{:03d}..".format(e+1, EPOCHS))

    # Training pass
    print("Training pass:")
    for data in tqdm(trainloader, total=len(trainloader)):
        images, labels = data[0].numpy(), data[1].numpy()

        prob, loss = model.forward(images, labels)
        model.backward(loss)
        optimizer.step()
        
        running_loss += np.sum(loss)
    
    # Testing pass
    print("Validation pass:")
    test_loss = 0
    accuracy = 0
    for data in tqdm(testloader, total=len(testloader)):
        images, labels = data[0].numpy(), data[1].numpy()
        
        prob, loss = model.forward(images, labels)
        test_loss += np.sum(loss)
        
        top_class = np.argmax(prob, axis=1)
        equals = (top_class == labels)
        accuracy += np.mean(equals.astype(float))

    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))
    accuracies.append(accuracy/len(testloader))
    
    epoch_times.append(time.process_time_ns())
    print("Training loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
          "Cur time(ns): {}".format(epoch_times[-1]))

#%% Evaluation

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label="Training loss")
ax.plot(test_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross Entropy Loss")
ax.legend()
ax2 = ax.twinx()
ax2.plot(np.array(accuracies)*100, label="Accuracy", color='g')
ax2.set_ylabel("Accuracy (%)")
plt.title("Training procedure")
plt.savefig(RESULT_PATH + 'training_proc.png', dpi = 100)

proc_results = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'epoch_times': epoch_times,
    'accuracies': accuracies,
}

# print(proc_results)
with open(RESULT_PATH + 'torch_results.json', 'w+') as f:
    json.dump(proc_results, f)
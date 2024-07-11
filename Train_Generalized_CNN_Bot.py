import copy
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from CNN_Generalized_Bot import Generalized_CNN
import torch.optim as optim
from torch import nn
from matplotlib import pyplot as plt
from Ship import get_ship
from sklearn.model_selection import train_test_split


def test_train_split(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    return torch.tensor(train_x), torch.tensor(test_x), torch.tensor(train_y), torch.tensor(test_y)


def load_process_training_data(train_data_path: str, random_seed: int):
    df = pd.read_csv(train_data_path)
    random.seed(random_seed)
    ship = get_ship()
    padded_ship = np.pad(ship, 1, 'constant', constant_values='#')
    df = df.dropna()
    train_x = df.drop('Optimal_Direction', axis=1).drop('Unnamed: 0', axis=1)
    train_y = df['Optimal_Direction']
    train_y = train_y.astype(
        pd.CategoricalDtype(categories=['STAY', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'NW', 'SW', 'SE', 'NE']))
    train_y = pd.get_dummies(train_y)
    train_x = torch.from_numpy(train_x.values).float()
    train_y = torch.from_numpy(train_y.values).float()
    tensor = torch.ones(())
    train_ship_x = tensor.new_empty(size=(train_x.shape[0], 5, 13, 13), dtype=float)
    for i in range(train_x.shape[0]):
        temp_ship = padded_ship.copy()
        bot_x, bot_y, crew_x, crew_y = train_x[i]
        temp_ship[int(bot_x.item()) + 1][int(bot_y.item()) + 1] = 'B'
        temp_ship[int(crew_x.item()) + 1][int(crew_y.item()) + 1] = 'C'
        df = pd.DataFrame(temp_ship.reshape((169)))
        df = df.astype(pd.CategoricalDtype(categories=['B', 'C', 'T', '#', 'O']))
        temp_ship_int = pd.get_dummies(df)
        temp_ship_int = temp_ship_int.values.reshape((13, 13, 5))
        temp_ship_int = np.transpose(temp_ship_int, (2, 0, 1))
        train_ship_x[i] = torch.tensor(temp_ship_int)
    return train_ship_x, train_y


def load_data_from_files(files: dict[str, int]):
    merged_train_x = None
    merged_train_y = None
    print('Collecting the data for all files and then merging')
    for file_name in files:
        print(f'Loading data from {file_name} file and processing it')
        train_x, train_y = load_process_training_data(file_name, 10 + files[file_name])
        print(f'Processed data from {file_name} file')
        if merged_train_x is None:
            merged_train_x = train_x
            merged_train_y = train_y
        else:
            merged_train_x = np.concatenate((merged_train_x, train_x), axis=0)
            merged_train_y = np.concatenate((merged_train_y, train_y), axis=0)
        print(f'Merged processed data from file {file_name}')
    return torch.tensor(merged_train_x), torch.tensor(merged_train_y)


def train(files):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generalized_CNN()
    model.to(device)
    print(f"Running training on: {device}")
    inputs, labels = load_data_from_files(files)
    train_x, test_x, train_y, test_y = test_train_split(inputs, labels)
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 2000
    print(f'Shape of train x is {train_x.shape}')
    print(f'Shape of train y is {train_y.shape}')
    print(f'Shape of test x is {test_x.shape}')
    print(f'Shape of test y is {test_y.shape}')
    losses = []
    accuracies = []
    max_accuracy = 0
    max_accuracy_epoch = -1
    best_model = None
    model.train()
    for i in range(epochs):
        start_time = time.time()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = loss_func(logits, train_y)
        probs = get_probs(logits)
        print(f'Epoch Number:{i}')
        print(f'Calculated loss between logits and train y:{loss.item()}')
        losses.append(loss.item())
        acc = torch.sum(torch.argmax(train_y, dim=1) == torch.argmax(probs, dim=1)) / train_y.shape[0]
        if acc > max_accuracy:
            max_accuracy = acc
            max_accuracy_epoch = i
            best_model = copy.deepcopy(model.state_dict())
        print(f'Accuracy:{acc}')
        accuracies.append(acc)
        loss.backward()
        optimizer.step()
        print(f'Completed epoch number {i} in {time.time() - start_time} seconds')
    print(f'Best accuracy achieved at {max_accuracy_epoch}th epoch and the accuracy is {max_accuracy}')
    torch.save(best_model,
               '/common/home/kd958/PycharmProjects/Robot-Guidance/best-CNN-Generalizing.pt')

    plot_loss_by_epochs(losses, 'training_losses.png')
    plot_loss_by_epochs(accuracies, 'training_accuracies.png')
    test_model(best_model, test_x, test_y)


def edit_wall_cells(ship):
    for i in range(len(ship)):
        for j in range(len(ship)):
            if i == 0 or j == 0 or i == len(ship) - 1 or j == len(ship) - 1:
                ship[i][j] = '#'
    return ship


def plot_loss_by_epochs(losses, file_name='plot.png'):
    # First, check if losses is a list and convert to tensor if it is
    if isinstance(losses, list):
        losses = torch.tensor(losses, dtype=torch.float32)  # Ensuring the data type is suitable for conversion

    # Move the tensor to CPU and convert to numpy for plotting
    losses = losses[2:].cpu().numpy()  # Skip the first two elements and convert

    # Create a range for the epochs based on the length of the adjusted losses
    epochs = [i for i in range(len(losses))]

    # Plotting the losses against the epochs
    plt.plot(epochs, losses)
    plt.savefig(file_name)
    plt.close()


def get_probs(logits):
    return nn.Softmax(dim=1)(logits)


def test_model(model_file, test_x, test_y):
    model = Generalized_CNN()
    model.load_state_dict(torch.load('/common/home/kd958/PycharmProjects/Robot-Guidance/best-CNN-Generalizing.pt'))
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    logits = model(test_x)
    probs = get_probs(logits)
    acc = torch.sum(torch.argmax(test_y, dim=1) == torch.argmax(probs, dim=1)) / test_y.shape[0]
    print(f'Accuracy achieved with the best model on test data is:{acc}')


# def process_data():

if __name__ == '__main__':
    files = {
        'Data_for_Generalizing_0.csv': 0,
        'Data_for_Generalizing_1.csv': 1,
        'Data_for_Generalizing_2.csv': 2,
        'Data_for_Generalizing_3.csv': 3,
        'Data_for_Generalizing_4.csv': 4,
        'Data_for_Generalizing_5.csv': 5,
        'Data_for_Generalizing_6.csv': 6,
        'Data_for_Generalizing_7.csv': 7,
        'Data_for_Generalizing_8.csv': 8,
        'Data_for_Generalizing_9.csv': 9
    }
    train(files)
    # test_model()

import copy
import random
import numpy as np
import pandas as pd
import torch
from CNNModel_OverFit import CNN_Model_Overfit
import torch.optim as optim
from torch import nn
from matplotlib import pyplot as plt
from Ship import get_ship


def load_process_training_data(train_data_path: str):
    df = pd.read_csv(train_data_path)
    random.seed(10)
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


def train(data_path):
    model = CNN_Model_Overfit()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    loss_func = nn.CrossEntropyLoss()
    train_x, train_y = load_process_training_data(data_path)
    epochs = 1200
    print(f'Shape of train x is {train_x.shape}')
    print(f'Shape of train y is {train_y.shape}')
    losses = []
    accuracies = []
    max_accuracy = 0
    max_accuracy_epoch = -1
    best_model = None
    for i in range(epochs):
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
    print(f'Best accuracy achieved at {max_accuracy_epoch}th epoch and the accuracy is {max_accuracy}')
    torch.save(best_model,
               '/Robot-Guidance/best-CNN-Overfit.pt')
    plot_loss_by_epochs(losses)
    plot_loss_by_epochs(accuracies)
    # test_model(best_model)


def plot_loss_by_epochs(losses):
    losses = losses[2:]
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.show()


def get_probs(logits):
    return nn.Softmax(dim=1)(logits)


def test_model():
    train_x, train_y = load_process_training_data('train_data.csv')
    model = CNN_Model_Overfit()
    model.load_state_dict(torch.load('/Robot-Guidance/best-CNN-Overfit.pt'))
    logits = model(train_x)
    probs = get_probs(logits)
    acc = torch.sum(torch.argmax(train_y, dim=1) == torch.argmax(probs, dim=1)) / train_y.shape[0]
    print(f'Accuracy achieved with the best model is:{acc}')

if __name__ == '__main__':
    train('train_data.csv')

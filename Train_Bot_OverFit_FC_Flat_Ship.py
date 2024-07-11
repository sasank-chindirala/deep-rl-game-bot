import copy
import random
import pandas as pd
import torch
from FC_Model_With_Flattened_Ship import FC_Model_With_Flattened_Ship
import torch.optim as optim
from torch import nn
from matplotlib import pyplot as plt
from Ship import get_ship

def load_process_training_data(train_data_path: str):
    df = pd.read_csv(train_data_path)
    map_cells = {
        'B': 3,
        'C': 2,
        'O': 1,
        '#': 0,
        'T': 1
    }
    random.seed(10)
    ship = get_ship()
    df = df.dropna()
    train_x = df.drop('Optimal_Direction', axis=1).drop('Unnamed: 0', axis=1)
    train_y = df['Optimal_Direction']
    train_y = train_y.astype(
        pd.CategoricalDtype(categories=['STAY', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'NW', 'SW', 'SE', 'NE']))
    train_y = pd.get_dummies(train_y)
    train_x = torch.from_numpy(train_x.values).float()
    train_y = torch.from_numpy(train_y.values).float()
    tensor = torch.ones(())
    train_ship_x = tensor.new_empty(size=(train_x.shape[0], 121), dtype=float)
    for i in range(train_x.shape[0]):
        temp_ship = ship.copy()
        bot_x, bot_y, crew_x, crew_y = train_x[i]
        temp_ship[int(bot_x.item())][int(bot_y.item())] = 'B'
        temp_ship[int(crew_x.item())][int(crew_y.item())] = 'C'
        temp_ship_int = [[map_cells[s] for s in temp_ship[i]] for i in range(len(temp_ship))]
        train_ship_x[i] = torch.tensor(temp_ship_int).flatten()
    return train_ship_x, train_y


def train(data_path):
    model = FC_Model_With_Flattened_Ship()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()
    train_x, train_y = load_process_training_data(data_path)
    epochs = 10000
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
               'C:/Users/harsh/OneDrive/Desktop/Rutgers/Sem1/Intro to AI/Project 3/Robot-Guidance/best-FC-Overfit.pt')

    plot_loss_by_epochs(losses)
    plot_loss_by_epochs(accuracies)


def edit_wall_cells(ship):
    for i in range(len(ship)):
        for j in range(len(ship)):
            if i == 0 or j == 0 or i == len(ship) - 1 or j == len(ship) - 1:
                ship[i][j] = '#'
    return ship


def plot_loss_by_epochs(losses):
    losses = losses[2:]
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.show()


def get_probs(logits):
    return nn.Softmax(dim=1)(logits)


def test_model():
    train_x, train_y = load_process_training_data('train_data.csv')
    model = FC_Model_With_Flattened_Ship()
    model.load_state_dict(torch.load('C:/Users/harsh/OneDrive/Desktop/Rutgers/Sem1/Intro to AI/Project 3/Robot-Guidance/best-FC-Overfit.pt'))
    logits = model(train_x)
    probs = get_probs(logits)
    acc = torch.sum(torch.argmax(train_y, dim=1) == torch.argmax(probs, dim=1)) / train_y.shape[0]
    print(f'Accuracy achieved with the best model is:{acc}')

if __name__ == '__main__':
    train('train_data.csv')

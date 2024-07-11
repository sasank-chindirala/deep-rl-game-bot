import copy
import itertools
import random
from tkinter import Tk, ttk
import numpy as np
import pandas as pd
import torch
from torch import nn
from ExpectedTimeNoBot import evaluate_expected_values
from Ship import get_ship


def show_tkinter(ship: np.ndarray):
    """
    :param ship: layout of the ship as a 2D matrix with each element representing whether the cell at that
                        coordinates is open/blocked/occupied by crew/teleport pad
    :return: None
    """
    root = Tk()
    table = ttk.Frame(root)
    table.grid()
    ship_dimension = len(ship)
    for row in range(ship_dimension):
        for col in range(ship_dimension):
            label = ttk.Label(table, text=ship[row][col], borderwidth=1, relief="solid")
            label.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
    root.mainloop()


def get_exp_time_no_bot_for_fixed_ship():
    random.seed(10)
    ship = get_ship()
    t_no_bot_grid = evaluate_expected_values(ship)


def generate_position_pairs(ship, k=100):
    open_cells = list(map(tuple, np.argwhere(ship == 'O')))
    position_pairs = []
    if len(open_cells) < 2:
        raise ValueError("Not enough open cells to generate pairs.")
    while len(position_pairs) < k:
        crew_pos, bot_pos = random.sample(open_cells, 2)
        position_pairs.append((crew_pos, bot_pos))
    return position_pairs


def run_simulation_for_learned_bot(model, ship, crew_position, bot_position):
    timestep = 0
    ship[crew_position[0], crew_position[1]] = 'C'
    ship[bot_position[0], bot_position[1]] = 'B'
    init_bot_posn = copy.deepcopy(bot_position)
    init_crew_posn = copy.deepcopy(crew_position)
    # print(f"Crew initial position: {crew_position}")
    # print(f"Bot initial position: {bot_position}")
    while crew_position != (5, 5):
        temp = bot_position
        ship, bot_position = bot_step(model, ship, bot_position, crew_position)
        ship, crew_position = crew_member_step(ship, bot_position, crew_position)
        if timestep > 2000:
            print(f"FAILURE: Couldn't reach teleport pad in 2000 steps within initial bot position: {init_bot_posn} and "
                  f"initial crew position as {init_crew_posn}")
            return timestep
        timestep += 1
    print(f"SUCCESS!: Time taken for crew to reach teleport pad for this configuration: {timestep};"
          f" initial bot position is {init_bot_posn} and initial crew position is {init_crew_posn}")
    return timestep


def run_simulation_for_learned_bot_for_k_positions_of_bot_and_crew(model, k):
    random.seed(10)
    ship = get_ship()
    posn_pairs = generate_position_pairs(ship, k)
    time_taken = []
    for i in range(k):
        ship = get_ship()
        crew_posn, bot_posn = posn_pairs[i]
        print(f'Running simulation number: {i}')
        time = run_simulation_for_learned_bot(model, ship, crew_posn, bot_posn)
        time_taken.append(time)
    time_taken = list(filter(lambda a: a != 2001, time_taken))
    success_rate = len(time_taken)/k
    print(f'Success rate for the bot when run for 1000 simulations is {success_rate}')
    average_time = np.mean(time_taken)
    print(f"Time taken for all pairs: {time_taken}")
    print(f"The average time for getting to teleport pad: {average_time}")


def generate_all_position_pairs(ship):
    open_cells = list(map(tuple, np.argwhere(ship == 'O')))
    position_pairs = list(itertools.combinations(open_cells, 2))

    return position_pairs


def tensorize_ship(ship: np.ndarray):
    padded_ship = np.pad(ship, 1, 'constant', constant_values='#')
    df = pd.DataFrame(padded_ship.reshape((-1, 1)), columns=['state'])
    df = df.astype(pd.CategoricalDtype(categories=['B', 'C', 'T', '#', 'O']))
    one_hot_encoded_ship = pd.get_dummies(df).to_numpy().reshape((13, 13, 5))
    ship_tensor = np.transpose(one_hot_encoded_ship, (2, 0, 1))
    input_tensor = torch.tensor(ship_tensor).unsqueeze(0).float()
    return input_tensor


def bot_step(model_path, ship, bot_position, crew_position):
    input_tensor = tensorize_ship(ship)
    input_tensor = input_tensor.float()
    action = predict_direction(input_tensor, model_path)
    nx, ny = bot_position[0] + action[0], bot_position[1] + action[1]
    if 0 <= nx < len(ship) and 0 <= ny < len(ship[0]) and ship[nx][ny] == 'O':  # Check if the new position is open
        ship[bot_position] = 'O'  # Set old position to open
        bot_new_posn = (nx, ny)
        ship[nx][ny] = 'B'  # Move bot to new position
    else:
        # print("Invalid move, hence staying in place!")
        bot_new_posn = bot_position  # Bot stays in place if move is invalid
    return ship, bot_new_posn


def crew_member_step(ship, bot_posn, crew_posn):
    crew_valid_next_positions = get_valid_crew_member_moves(crew_posn, bot_posn, ship)
    if len(crew_valid_next_positions) == 0:
        return ship, crew_posn
    next_crew_posn = random.choice(crew_valid_next_positions)
    ship[next_crew_posn[0]][next_crew_posn[1]] = 'C'
    ship[crew_posn[0]][crew_posn[1]] = 'O'
    return ship, next_crew_posn


def get_valid_crew_member_moves(crew_posn, bot_posn, ship):
    ship_dim = len(ship)
    x, y = crew_posn
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    valid_next_positions = []
    for dx, dy in directions:
        if 0 <= x + dx < ship_dim and 0 <= y + dy < ship_dim and ship[x + dx][y + dy] != '#':
            valid_next_positions.append((x + dx, y + dy))
    if bot_posn in valid_next_positions:
        valid_next_positions.remove(bot_posn)
    return valid_next_positions


def predict_direction(data, model):
    logits = model(data)
    probs = nn.Softmax(dim=1)(logits)
    predicted_label = torch.argmax(probs, dim=1)
    possible_directions = ['STAY', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'NW', 'SW', 'SE', 'NE']
    directions = {
        'STAY': (0, 0),
        'RIGHT': (0, 1),
        'LEFT': (0, -1),
        'DOWN': (1, 0),
        'UP': (-1, 0),
        'NW': (-1, -1),
        'SW': (1, -1),
        'SE': (1, 1),
        'NE': (-1, 1)
    }
    return directions[possible_directions[predicted_label]]


if __name__ == '__main__':
    get_exp_time_no_bot_for_fixed_ship()
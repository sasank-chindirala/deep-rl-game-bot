import random
import numpy as np


def get_ship():
    random.seed(10)
    blocked_cells = [(4, 4), (4, 6), (6, 4), (6, 6)]
    ship = np.full((11, 11), 'O')
    ship[5][5] = 'T'
    open_cells = np.where(ship == 'O')
    open_cells_list = list(zip(open_cells[0], open_cells[1]))
    cells_not_to_be_blocked = [(3,5), (4, 5), (6, 5), (7, 5), (5, 3), (5, 4), (5, 6), (5, 7)]
    blockable_cells = [cell for cell in open_cells_list if cell not in cells_not_to_be_blocked]
    blocked_cells += random.sample(blockable_cells, 10)
    for pos in blocked_cells:
        ship[pos[0]][pos[1]] = '#'
    return ship



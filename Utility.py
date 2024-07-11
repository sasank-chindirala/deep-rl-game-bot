import math


def is_neighbor(cell1: tuple[int, int], cell2: tuple[int, int]):
    x, y = cell1
    x1, y1 = cell2
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dx = x1 - x
    dy = y1 - y
    if (dx, dy) in directions:
        return True
    return False


def get_num_open_neighbors(position: tuple[int, int], ship_layout: list[list[str]]) -> float:
    # Define the possible directions (up, down, left, right)
    ship_dim = len(ship_layout)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors: list[tuple[int, int]] = []
    for dx, dy in directions:
        neighbor_x, neighbor_y = position[0] + dx, position[1] + dy
        if 0 <= neighbor_x < ship_dim and 0 <= neighbor_y < ship_dim and ship_layout[neighbor_x][neighbor_y] != '#':
            neighbors.append((neighbor_x, neighbor_y))
    return len(neighbors)


def de_vectorize_index_to_2D(vectorized_index: int, dim: int = 11) -> tuple[int, int]:
    return vectorized_index // dim, vectorized_index % dim


def de_vectorize_index_to_4D(vectorized_index: int, dim: int = 11) -> tuple[int, int, int, int]:
    i1 = int(vectorized_index // math.pow(dim,3))
    rem = vectorized_index % math.pow(dim,3)
    j1 = int(rem // math.pow(dim,2))
    rem = rem % math.pow(dim,2)
    i2 = int(rem // dim)
    j2 = int(rem % dim)
    return i1,j1,i2,j2


def get_manhattan_dist(cell1: tuple[int, int], cell2: tuple[int, int]):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

def get_vectorized_index_from_4D(indexes:tuple[int,int,int,int],dim=11):
    return int(math.pow(dim,3)*indexes[0] + math.pow(dim,2)*indexes[1] + dim * indexes[2] + indexes[3])

# def

import random
import time

from Ship import get_ship
from ExpectedTimeWithBot import policy_iteration, save_policy_to_csv


class DataGenerator:
    def __init__(self, num_ship_layouts):
        print('Initializing data generator object')
        self.num_ship_layouts = num_ship_layouts

    def generate_data(self):
        for i in range(self.num_ship_layouts):
            start_time = time.time()
            random_seed = 10 + i
            file_name = 'Data_for_Generalizing_' + str(i) + '.csv'
            random.seed(random_seed)
            print(f'Generating {i}th ship with random seed set as {random_seed}')
            ship = get_ship()
            print(f'Generated {i}th ship with random seed set as {random_seed}')
            print(f'Generating optimal policy for {i}th ship')
            policy_directions = policy_iteration(ship)
            print(f'Generated optimal policy. Saving it to file {file_name}')
            save_policy_to_csv(policy_directions, file_name)
            print(
                f'Generated optimal policy for {i}th policy and save to file {file_name} in {time.time() - start_time}'
                f'seconds')

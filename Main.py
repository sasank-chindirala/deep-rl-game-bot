import argparse

import torch

from CNN_Generalized_Bot import Generalized_CNN
from ExpectedTimeNoBot import evaluate_expected_values, get_expected_time
from Ship import get_ship
from Simulation import run_simulation_for_learned_bot, \
    run_simulation_for_learned_bot_for_k_positions_of_bot_and_crew, show_tkinter
from DataGenerator import DataGenerator


def main():
    model_path = ('C:/Users/harsh/OneDrive/Desktop/Rutgers/Sem1/Intro to AI/Project 3/Robot-Guidance/best-CNN-Generalizing.pt')
    model = Generalized_CNN()
    model.load_state_dict(torch.load(model_path))
    model = model.float()  # Ensure the model is using float32
    model.eval()  # Switch the model to evaluation mode
    run_simulation_for_learned_bot_for_k_positions_of_bot_and_crew(model, 1000)
    # data_generator = DataGenerator(10)
    # data_generator.generate_data()
    # ship = get_ship()
    # print(get_expected_time(evaluate_expected_values(ship)))


if __name__ == '__main__':
    main()

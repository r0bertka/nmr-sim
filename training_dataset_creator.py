import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

freq_range = np.arange(start=2.85, stop=2.89, step=2e-4)
baseline = 100
height0 = 55
central_frequency = 2.87
broadening = 400e-6
noise_level_percentage = 5

number_of_datasets = 1

nuclear_couplings = [13.72e-3, 12.78e-3, 8.92e-3, 6.52e-3, 4.20e-3, 2.40e-3]


@dataclass
class Splitting:
    multiplicity: int
    delta_f: float


def calc_lorentzian(frequency: np.ndarray, central_frequency: float,
                    height: float, broadening: float):
    return height * broadening ** 2 / (
            (frequency - central_frequency) ** 2 + broadening ** 2)


def split(central_frequency: float, delta_f: float, multiplicity: int) -> list[
    float]:
    factors = [-(multiplicity - 1) / 2 + _ for _ in range(multiplicity)]
    return [central_frequency + factor * delta_f for factor in factors]


def calc_height(splittings, height) -> float:
    factor_old = 1
    factor_total = 1
    for splitting in splittings:
        factor_total = factor_old * splitting.multiplicity
        factor_old = factor_total
    return height / factor_total


def split_wrap(splittings, freqs: list, i=0):
    if i == len(splittings) - 1:
        all_freqs = []
        for f in freqs:
            all_freqs.extend(
                split(f, splittings[i].delta_f, splittings[i].multiplicity))
        return all_freqs

    temp_freqs = []
    for f in freqs:
        temp_freqs.extend(
            split(f, splittings[i].delta_f, splittings[i].multiplicity))
    return split_wrap(splittings, temp_freqs, i + 1)


nitrogen_splitting = Splitting(3, 2.12e-3)

for i in range(number_of_datasets):
    two_random_carbons = np.random.randint(0, 6, 2)
    carbon_splitting1 = Splitting(2, nuclear_couplings[two_random_carbons[0]])
    carbon_splitting2 = Splitting(2, nuclear_couplings[two_random_carbons[1]])
    splittings = (nitrogen_splitting,
                  carbon_splitting1,
                  carbon_splitting2)
    all_freqs = split_wrap(splittings, [central_frequency], i=0)
    new_height = calc_height(splittings, height0)
    lorentzians = np.zeros(len(freq_range))

    for freq in all_freqs:
        lorentzians += calc_lorentzian(freq_range, freq, new_height, broadening)

    spectrum = baseline - lorentzians
    noise = -noise_level_percentage / 2 + noise_level_percentage * np.random.random(len(freq_range))
    dataset = [f'{round(freq_range[_], 4)} \t {spectrum[_] + noise[_]}' for _ in range(len(freq_range))]

    file_name = f'training_dataset_spectrum_{i}.dat'
    with open(f'training\\{file_name}', 'w') as file:
        file.write(f'Carbon 1: {min(round(carbon_splitting1.delta_f * 1e3, 2), round(carbon_splitting2.delta_f * 1e3, 2))} MHz\n')
        file.write(f'Carbon 2: {max(round(carbon_splitting1.delta_f * 1e3, 2), round(carbon_splitting2.delta_f * 1e3, 2))} MHz\n')
        file.write('\n')
        file.write('frequency (GHz) \t norm. intensity (arb. u.)\n')
        for line in dataset:
            file.write(f'{line}\n')

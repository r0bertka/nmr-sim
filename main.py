import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def calc_lorentzian(frequency: np.ndarray, central_frequency: float, height: float, broadening: float):
    return height * broadening **2 / ((frequency - central_frequency) **2 + broadening **2)

import itertools
freq_range = np.arange(start=2.83, stop=2.91, step=1e-5)
baseline = 100
height0 = 20
central_frequency = 2.87
broadening = 500e-6


def split(central_frequency: float, delta_f: float, multiplicity: int) -> list[float]:
    factors = [-(multiplicity - 1) / 2 + _ for _ in range(multiplicity)]
    return [central_frequency + factor * delta_f for factor in factors]

def calc_height(splittings, height) -> float:
    factor_old = 1
    factor_total = 1
    for splitting in splittings:
        factor_total = factor_old * splitting.multiplicity
        factor_old = factor_total
    return height / factor_total

@dataclass
class Splitting:
    multiplicity: int
    delta_f: float


nitrogen_splitting = Splitting(3, 2.12e-3)
carbon1_splitting = Splitting(2, 4e-3)
carbon2_splitting = Splitting(2, 14.0e-3)
carbon3_splitting = Splitting(2, 5e-3)

splittings = (
    nitrogen_splitting,
    carbon1_splitting,
    carbon2_splitting,
    carbon3_splitting
)


def split_wrap(freqs: list, i=0):
    if i == len(splittings) - 1:
        all_freqs = []
        for f in freqs:
            all_freqs.extend(split(f, splittings[i].delta_f, splittings[i].multiplicity))
        return all_freqs

    temp_freqs = []
    for f in freqs:
        temp_freqs.extend(split(f, splittings[i].delta_f, splittings[i].multiplicity))
    return split_wrap(temp_freqs, i + 1)


all_freqs = split_wrap([central_frequency], i=0)
new_height = calc_height(splittings, height0)

lorentzians = np.zeros(len(freq_range))

for freq in all_freqs:
    lorentzians += calc_lorentzian(freq_range, freq, new_height, broadening)

spectrum = baseline - lorentzians

plt.plot(freq_range, spectrum)
# for i in range(len(carbon_splitted_frequencies)):
#     plt.plot(freq_range, lorentzians[f'lorentzian{i}'])
plt.show()


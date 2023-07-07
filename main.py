import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def calc_lorentzian(frequency: np.ndarray, central_frequency: float, height: float, broadening: float):
    return height * broadening ** 2 / ((frequency - central_frequency) ** 2 + broadening ** 2)


freq_range = np.arange(start=2.85, stop=2.89, step=1e-5)
baseline = 100
height0 = 55
central_frequency = 2.87
broadening = 300e-6


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
#  double_NV_splitting = Splitting(2, 200e-6)
#  nitrogen_splitting = Splitting(2, 3.0e-3)
carbon1_splitting = Splitting(2, 3.3e-3)
carbon2_splitting = Splitting(2, 8.7e-3)
# carbon3_splitting = Splitting(2, 2.4e-3)
# carbon4_splitting = Splitting(2, 2.4e-3)

splittings = (
#      double_NV_splitting,
    nitrogen_splitting,
    carbon1_splitting,
    carbon2_splitting,
    #carbon3_splitting,
   # carbon4_splitting
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
# plt.savefig('C:/Users/rober/Documents/Aufbau_QC/Pulsed tests/simulierte NMR-Spektren/8.9+2x0.9MHz.png',
#               format='png')

plt.show()

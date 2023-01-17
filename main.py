import numpy as np
import matplotlib.pyplot as plt

def Lorentzian(frequency: np.ndarray, central_frequency: float, height: float, broadening: float):
    return height * broadening **2 / ((frequency - central_frequency) **2 + broadening **2)

import itertools
freq_range = np.arange(start=2.83, stop=2.91, step=1e-4)
baseline = 100
height0 = 20
central_frequency0 = 2.87
broadening = 500e-6

Lorentzian0 = Lorentzian(frequency=freq_range, central_frequency=central_frequency0, height=height0, broadening=broadening)

nitrogen_delta_f = [2.2e-3]
carbon_delta_f = [2.2e-3, 2.2e-3]
plitted_frequencies = []
#  carbon_splitted_frequencies = []
final_frequencies = []
for nitrogen_splitting in nitrogen_delta_f:
    splitted_frequencies = [central_frequency0 - nitrogen_splitting,
                                     central_frequency0,
                                     central_frequency0 + nitrogen_splitting
                                     ]

for frequency in splitted_frequencies:
    for carbon_splitting in carbon_delta_f:
       final_frequencies.append(frequency)
    # for fn, fc in itertools.product(nitrogen_splitted_frequencies, carbon_delta_f):
    #     carbon_splitted_frequencies.extend((fn - fc/2, fn + fc/2))
print(splitted_frequencies[:])

lorentzians = {}

for i in range(len(carbon_splitted_frequencies)):
    rescaled_height = height0 / len(carbon_splitted_frequencies)
    lorentzians[f'lorentzian{i}'] = Lorentzian(freq_range,
                                               carbon_splitted_frequencies[i],
                                               rescaled_height,
                                               broadening)

#  print(lorentzians['lorentzian1'])

for i in range(len(carbon_splitted_frequencies)):
    plt.plot(freq_range, lorentzians[f'lorentzian{i}'])
plt.show()


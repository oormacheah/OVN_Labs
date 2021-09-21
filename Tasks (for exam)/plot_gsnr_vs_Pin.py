from pathlib import Path
from core.elements import Network
from core.elements import Line
import matplotlib.pyplot as plt
from numpy import arange
from core.conversions import lin2db, path_str2arrow

# Use elements.py from Lab 8's latest commit


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'
    n_channels = 10
    network = Network(json_path, n_channels)
    network.connect()

    line_label = 'AD'  # Insert generic line here

    gsnrs = []
    signal_powers = arange(0.0001, 0.0015, 0.00001)
    for power in signal_powers:
        line_dict = network.lines
        noise = line_dict[line_label].noise_generation(power)
        gsnrs.append(lin2db(power / noise))

    plt.figure()
    plt.plot(signal_powers, gsnrs)
    plt.xlabel('Input power [W]')
    plt.ylabel('GSNRi [dB]')
    plt.title(f'GSNRi vs Pin ({line_label})')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

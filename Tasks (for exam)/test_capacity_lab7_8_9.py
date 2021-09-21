from pathlib import Path
import matplotlib.pyplot as plt
import random
from core.elements import Network
from core.elements import Connection
from si_prefix import si_format

# Use elements.py from last commit of Lab 7, then Lab 8 (for showing difference in performance) and finally the
# 'presentable_commit' for testing LOGO strategy of Lab 9

# Change the default transceiver in the used elements.py, since the exam json file doesn't provide it


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'

    input_signal_power = 1e-3
    n_connections = 100
    n_channels = 10
    network = Network(json_path, n_channels)
    network.connect()
    network.weighted_paths = network.weigh_paths(input_signal_power)

    node_labels = list(network.nodes.keys())
    x_sample = []
    connection_list = []

    for i in range(n_connections):  # Generate the random connections
        rand_pair = random.sample(node_labels, 2)
        connection_list.append(Connection(rand_pair[0], rand_pair[1], input_signal_power))
        x_sample.append(int(i))

    network.stream(connection_list, 'snr')  # You can also choose 'latency'
    print('updated route space:\n', network.route_space)
    print(f'\n{network.weighted_paths}\n')

    bit_rates = []
    tot_capacity = 0

    rejection_count = 0
    for connection in connection_list:
        if connection.snr == 0.0:
            rejection_count += 1

    print('\nRejected connection count:', rejection_count)

    for connection in connection_list:
        bit_rates.append(connection.bit_rate)
        tot_capacity += connection.bit_rate

    avg_bit_rate = tot_capacity / len(connection_list)
    print(f'\nAvg over {n_connections} connections: {si_format(avg_bit_rate, 4)}bps')
    print(f'Total capacity deployed: {si_format(tot_capacity, precision=4)}bps')

    # Plotting
    plt.figure()
    plt.hist(bit_rates)
    plt.xlabel('Rb [bps]')
    plt.ylabel('Coincidences')
    plt.grid(True)
    plt.title('Bit Rate distribution')
    plt.show()


if __name__ == '__main__':
    main()

from si_prefix import si_format
from pathlib import Path
import matplotlib.pyplot as plt
import random
from core.elements import Network
from core.elements import Connection
from statistics import stdev, mean
from core.conversions import lin2db, db2lin

# Use elements.py from last commit of Lab 5

# To emulate first lab 4 test (100 connections without considering occupancy of a channel), a very big number of
# channels per line will be assumed

# File tests connections in the network, assuming only 1 channel per line and reporting the weighted paths, routing
# space and distributions of SNR and latencies (For lab 4 second test)


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'

    input_signal_power = 1
    n_channels = 10
    n_connections = 100
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

    network.stream(connection_list, 'latency')  # You can also choose 'latency'

    latencies = []
    snrs = []

    print('Weighted paths (1W input signal):')
    print(network.weighted_paths, '\n')
    print('Route space:')
    print(network.route_space, '\n')

    rejection_count = 0
    for connection in connection_list:
        if connection.snr == 0.0:
            rejection_count += 1
            print(f'connection REJECTED: {connection.input} -> {connection.output}')
        else:
            print(f'connection accepted: {connection.input} -> {connection.output}\t'
                  f'SNR: {si_format(connection.snr, 4)}dB\tLatency: {si_format(connection.latency, 4)}s')
            latencies.append(connection.latency)
            snrs.append(connection.snr)

    snrs_lin = [db2lin(snr) for snr in snrs]  # list of the snrs in linear (to compute mean and std)

    # Print mean and std
    print(f'\nGSNR\t\tMean: {si_format(lin2db(mean(snrs_lin)), 4)}dB\t'
          f'STD: {si_format(lin2db(stdev(snrs_lin)), 4)}dB')
    print(f'Latencies\tMean: {si_format(mean(latencies), 4)}s\t\tSTD: {si_format(stdev(latencies), 4)}s')
    print('Rejected connection count:', rejection_count)

    # Plotting
    plt.figure('Latency distribution')
    plt.hist(latencies)
    plt.xlabel('Latency')
    plt.grid()
    plt.title('Latency distribution')

    plt.figure('GSNR distribution')
    plt.hist(snrs)
    plt.xlabel('GSNR [dB]')
    plt.grid()
    plt.title('GSNR distribution')

    plt.show()


if __name__ == '__main__':
    main()

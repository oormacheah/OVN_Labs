from si_prefix import si_format
from pathlib import Path
import matplotlib.pyplot as plt
from core.elements import Network
from core.conversions import db2lin, lin2db
from numpy import arange

# Use elements.py from 'presentable_commit'

# Plot Average and Total capacity deployed vs. increasing M, reinitializing the network at each iteration of M


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'  # Insert the file to evaluate

    max_m = 30
    block_event_limit = 100
    n_channels = 10
    input_signal_power = 1  # Doesn't matter since the optimized launch power will be used
    tot_capacity_list = []
    avg_rb_list = []

    for m in range(1, max_m):  # Call the initialization from 0 of the network all over again
        network = Network(json_path, n_channels)
        network.connect()
        network.weighted_paths = network.weigh_paths(input_signal_power)  # Watts

        node_labels = list(network.nodes.keys())

        traffic_matrix = []  # Initialize as list of lists

        number_node_dict = dict(enumerate(network.nodes.keys()))
        # Created dictionary for row, column correspondence to source, destination nodes

        # Create matrix each iteration until max M
        for i in range(len(node_labels)):
            traffic_columns = []
            for j in range(len(node_labels)):
                if i != j:
                    if network.find_paths(number_node_dict[i],
                                          number_node_dict[j]):  # If there is a path for those nodes
                        traffic_columns.append(m * 100e9)
                    else:
                        traffic_columns.append(0.0)
                else:
                    traffic_columns.append(0.0)
            traffic_matrix.append(traffic_columns)

        results = network.traffic_matrix_deployment(traffic_matrix, block_event_limit)
        attempted_connection_list = results[0]

        tot_capacity = 0

        for connection in attempted_connection_list:
            tot_capacity += connection.bit_rate

        tot_capacity_list.append(tot_capacity)

        avg_bit_rate_a = tot_capacity / len(attempted_connection_list)
        avg_rb_list.append(avg_bit_rate_a)

        print(f'\nM = {m}')
        print(f'\nTotal capacity deployed: {si_format(tot_capacity, precision=4)}bps')
        print(f'Avg. Rb over {len(attempted_connection_list)} attempted connections: {si_format(avg_bit_rate_a, 4)}bps')

    # Plotting
    plt.figure()
    plt.plot(arange(1, max_m, 1), tot_capacity_list, marker='o')
    plt.xlabel('M value')
    plt.ylabel('Total deployed capacity [bps]')
    plt.grid(True)
    plt.title('Total capacity vs. M')

    plt.figure()
    plt.plot(arange(1, max_m, 1), avg_rb_list, marker='o')
    plt.xlabel('M value')
    plt.ylabel('Average deployed capacity [bps]')
    plt.grid(True)
    plt.title('Average capacity vs. M')
    plt.show()


if __name__ == '__main__':
    main()

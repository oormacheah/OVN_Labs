from si_prefix import si_format
from pathlib import Path
import matplotlib.pyplot as plt
from core.elements import Network
from core.conversions import db2lin, lin2db

# Use elements.py from 'presentable_commit'


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'  # Insert the file to evaluate

    block_event_limit = 100
    n_channels = 10
    input_signal_power = 1  # Doesn't matter since the optimized launch power will be used
    network = Network(json_path, n_channels)
    network.connect()
    network.weighted_paths = network.weigh_paths(input_signal_power)  # Watts

    node_labels = list(network.nodes.keys())

    m = 4
    traffic_matrix = []  # Initialize as list of lists

    number_node_dict = dict(enumerate(network.nodes.keys()))
    # Created dictionary for row, column correspondence to source, destination nodes

    for i in range(len(node_labels)):
        traffic_columns = []
        for j in range(len(node_labels)):
            if i != j:
                if network.find_paths(number_node_dict[i], number_node_dict[j]):  # If there is a path for those nodes
                    traffic_columns.append(m * 100e9)
                else:
                    traffic_columns.append(0.0)
            else:
                traffic_columns.append(0.0)
        traffic_matrix.append(traffic_columns)

    results = network.traffic_matrix_deployment(traffic_matrix, block_event_limit)
    attempted_connection_list = results[0]
    n_successful_connections = results[1]

    bit_rates = []
    tot_capacity = 0
    tot_gsnr = 0

    for connection in attempted_connection_list:
        bit_rates.append(connection.bit_rate)
        tot_capacity += connection.bit_rate
        tot_gsnr += db2lin(connection.snr)

    avg_bit_rate_a = tot_capacity / len(attempted_connection_list)
    avg_gsnr_a = lin2db(tot_gsnr / len(attempted_connection_list))
    avg_bit_rate_s = tot_capacity / n_successful_connections
    avg_gsnr_s = lin2db(tot_gsnr / n_successful_connections)
    print(f'\nM = {m}')
    print(f'\nTotal capacity deployed: {si_format(tot_capacity, precision=4)}bps')
    print(f'\nAvg. Rb over {n_successful_connections} successful connections: {si_format(avg_bit_rate_s, 4)}bps')
    print(f'Avg. Rb over {len(attempted_connection_list)} attempted connections: {si_format(avg_bit_rate_a, 4)}bps')
    print(f'\nAvg. GSNR over {n_successful_connections} successful connections: {si_format(avg_gsnr_s, 4)}dB')
    print(f'Avg. GSNR over {len(attempted_connection_list)} attempted connections: {si_format(avg_gsnr_a, 4)}dB')

    # Plotting
    plt.figure()
    plt.hist(bit_rates)
    plt.xlabel('Rb [bps]')
    plt.ylabel('Occurrences')
    plt.grid(True)
    plt.title('Bit Rate occurrences')
    plt.show()


if __name__ == '__main__':
    main()

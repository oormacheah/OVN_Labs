from si_prefix import si_format
from pathlib import Path
import matplotlib.pyplot as plt
from core.elements import Network
from core.conversions import db2lin, lin2db


def main():

    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / 'nodes_full_flex_rate.json'

    block_event_limit = 100
    input_signal_power = 1
    network = Network(json_path)
    network.connect()
    network.weighted_paths = network.weigh_paths(input_signal_power)  # Watts

    node_labels = list(network.nodes.keys())

    m = 6
    traffic_matrix = []  # Initialize as list of lists
    for i in range(len(node_labels)):
        traffic_columns = []
        for j in range(len(node_labels)):
            if i != j:
                traffic_columns.append(m * 100e9)
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
    print(f'\nTotal capacity used: {si_format(tot_capacity, precision=4)}bps')
    print(f'\nAvg. Rb over {n_successful_connections} successful connections: {si_format(avg_bit_rate_s, 4)}bps')
    print(f'Avg. Rb over {len(attempted_connection_list)} attempted connections: {si_format(avg_bit_rate_a, 4)}bps')
    print(f'\nAvg. GSNR over {n_successful_connections} successful connections: {si_format(avg_gsnr_s, 4)}')
    print(f'Avg. GSNR over {len(attempted_connection_list)} attempted connections: {si_format(avg_gsnr_a, 4)}')

    # Plotting
    plt.figure()
    plt.hist(bit_rates)
    plt.xlabel('Rb')
    plt.ylabel('Occurrences')
    plt.grid(True)
    plt.title('Bit Rate occurrences')
    plt.show()


if __name__ == '__main__':
    main()

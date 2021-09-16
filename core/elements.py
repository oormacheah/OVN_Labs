from pandas import DataFrame
from core.conversions import lin2db, db2lin, path_str2arrow, path_arrow2str
from scipy.special import erfcinv
import json
from math import sqrt, log10, log2, floor
from scipy.constants import c, h, e, pi
import matplotlib.pyplot as plt
from numpy import array, zeros, ones, cbrt
import random


class SignalInformation:
    def __init__(self, s_power, path):
        self.signal_power = s_power
        self.noise_power = 0.0
        self.latency = 0.0
        self.path = path

    def add_signal_power(self, increase_s):
        self.signal_power += increase_s

    def add_noise_power(self, increase_n):
        self.noise_power += increase_n

    def add_latency(self, inc_latency):
        self.latency += inc_latency

    def path_update(self):
        self.path = self.path[1:]


class Node:
    def __init__(self, node_dict):
        self.label = node_dict['label']
        self.position = node_dict['position']
        self.connected_nodes = node_dict['connected_nodes']
        self.successive = dict()
        self.switching_matrix = None
        if 'transceiver' in node_dict:
            self.transceiver = node_dict['transceiver']
        else:
            self.transceiver = 'fixed_rate'

    def propagate(self, signal_information):
        path = signal_information.path
        # if there are still elements to reach in the path list
        if len(signal_information.path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.signal_power = line.optimized_launch_power()
            signal_information.path_update()  # delete path[0]
            signal_information = line.propagate(signal_information)
        return signal_information


class Line:
    def __init__(self, line_dict, n_channels=10):
        self.label = line_dict['label']
        self.length = line_dict['length']
        self.successive = {}
        self.n_channels = n_channels
        self.state = ['free' for i in range(self.n_channels)]
        self.n_amplifiers = floor(self.length / 80e3)
        self.gain = 16  # For each amplifier (in dB)
        self.noise_figure = 3  # For each amplifier (in dB)  (try 5dB for Lab9)

        # Fiber parameters
        self.alpha_db = -0.2e-3  # [dB/m]
        self.beta2 = 2.13e-26  # [m Hz^2]^-1  (try 0.6e-26 for Lab9)
        self.gamma = 1.27e-3  # [m W]^-1
        self.rs = 32e9  # Symbol rate [Hz]
        self.df = 50e9  # Spacing between consecutive channels [Hz]
        self.alpha = self.alpha_db / (20 * log10(e))
        self.l_eff = 1 / (2 * self.alpha)

    def latency_generation(self):
        # propagation velocity (2/3 c)
        latency = self.length / (2 / 3 * c)
        return latency

    def noise_generation(self, signal_power):
        noise_ase = self.ase_generation()
        noise_nli = self.nli_generation(signal_power)
        noise = noise_ase + noise_nli
        return noise

    def ase_generation(self):
        f = 193.414e12  # C-band center
        bn = 12.5e9  # Noise bandwidth
        nf_linear = db2lin(self.noise_figure)
        g = db2lin(self.gain)
        ase = self.n_amplifiers * (h * f * bn * nf_linear * (g - 1))
        return ase

    def nli_generation(self, power_per_channel):
        bn = 12.5e9  # Noise bandwidth
        eta_nli = ((16 / (27 * pi)) * log10(((pi ** 2) / 2) *
                   ((self.beta2 * (self.rs ** 2)) / self.alpha) *
                   (self.n_channels ** (2 * (self.rs / self.df)))) *
                   ((self.gamma ** 2) / (4 * self.alpha * self.beta2)) * (1 / (self.rs ** 3)))
        nli = (power_per_channel ** 3) * eta_nli * self.n_amplifiers * bn
        return nli

    def propagate(self, signal_information):
        # add latency
        latency = self.latency_generation()  # generated the latency (proper for the line)
        signal_information.add_latency(latency)

        # add ASE and NLI
        signal_power = signal_information.signal_power
        noise_power = self.noise_generation(signal_power)
        signal_information.add_noise_power(noise_power)

        # propagate (recursive)
        if isinstance(signal_information, LightPath):  # weigh_paths method uses SignalInformation class
            self.state[signal_information.channel] = 'occupied'
        node_label = self.successive[signal_information.path[0]]
        signal_information = node_label.propagate(signal_information)
        return signal_information

    def optimized_launch_power(self):
        eta_nli = ((16 / (27 * pi)) * log10(((pi ** 2) / 2) *
                                            ((self.beta2 * (self.rs ** 2)) / self.alpha) *
                                            (self.n_channels ** (2 * (self.rs / self.df)))) *
                   ((self.gamma ** 2) / (4 * self.alpha * self.beta2)) * (1 / (self.rs ** 3)))
        nf = db2lin(self.noise_figure)  # in linear
        f0 = 193.414e12  # C-band frequency
        loss = self.alpha * self.length
        p_opt = cbrt((nf * loss * h * f0) / (2 * eta_nli))
        return p_opt


class Network:
    def __init__(self, json_path, n_channels=10):
        with open(json_path, 'r') as f:
            json_nodes = json.load(f)
        self.nodes = dict()
        self.lines = dict()
        self.weighted_paths = DataFrame()
        self.route_space = DataFrame()
        self.channels = n_channels  # for modularity

        for node_key in json_nodes:
            # instantiate nodes
            node_dict = json_nodes[node_key]  # each json_node_key value is a dict (containing pos and connected nodes)
            node_dict['label'] = node_key  # manually add key-value pair because it's not in json file
            self.nodes[node_key] = Node(node_dict)
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                # line label instantiate
                line_label = node_key + connected_node_label  # main (for) loop key + inner (for) loop key
                line_dict['label'] = line_label

                # position: (x, y) tuple
                node_position = node_dict['position']
                connected_node_position = json_nodes[connected_node_label]['position']
                x1 = node_position[0]
                y1 = node_position[1]
                x2 = connected_node_position[0]
                y2 = connected_node_position[1]
                line_dict['length'] = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                self.lines[line_label] = Line(line_dict, n_channels)  # channels as extra argument

        # self.connect()  # Constructor has to connect the network otherwise dataframe won't be able to be created
        # (Because it has to construct the weighted paths)

        # Initialize weighted paths
        # self.weighted_paths = self.weigh_paths(1e-3)  # Default signal power from Lab3 with 1mW sample signal

        #  Initialize route_space
        all_pairs = []
        all_paths = []
        node_dict = self.nodes
        for label1 in node_dict:
            for label2 in node_dict:
                if label1 != label2:
                    all_pairs.append(label1 + label2)  # Generating every possible pair
        for pair in all_pairs:
            possible_paths = self.find_paths(pair[0], pair[1])
            for path in possible_paths:
                all_paths.append(path)

        channel_list_of_dicts = []
        index_list = []

        for path in all_paths:
            channel_dict = dict()
            for j in range(self.channels):
                channel_dict[j] = 'free'
            index_list.append(path)
            channel_list_of_dicts.append(channel_dict)

        self.route_space = DataFrame(channel_list_of_dicts, index_list)

    # END OF CONSTRUCTOR

    # METHODS

    def connect(self):  # method to initialize successive elements in lines and nodes
        # take nodes and lines dictionaries from self to iterate over them
        node_dict = self.nodes
        line_dict = self.lines
        for node_label in node_dict:
            node = node_dict[node_label]
            for connected_node_label in node.connected_nodes:
                line_label = node_label + connected_node_label
                line = line_dict[line_label]
                line.successive[connected_node_label] = node_dict[connected_node_label]
                node.successive[line_label] = line_dict[line_label]

        # Now it also initializes the switching matrix
        switching_matrix = dict()
        for input_node in node_dict:
            input_node_dict = dict()
            for output_node in node_dict:
                if input_node == output_node:
                    input_node_dict[output_node] = zeros(self.channels)
                else:
                    input_node_dict[output_node] = ones(self.channels)
            switching_matrix[input_node] = input_node_dict
            self.nodes[input_node].switching_matrix = switching_matrix

    def find_paths(self, label1, label2):  # find all possible paths from one node1 to node2 passing max 1 time per node
        available_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]  # crossable nodes
        available_lines = self.lines.keys()  # all the lines
        possible_paths = {'0': label1}
        for i in range(len(available_nodes)):
            possible_paths[str(i + 1)] = []
            for inner_path in possible_paths[str(i)]:
                for cross_node in available_nodes:
                    if ((inner_path[-1] + cross_node) in available_lines) & (cross_node not in inner_path):
                        possible_paths[str(i + 1)] += [inner_path + cross_node]  # possible_paths[] is a list and we're
                        # adding current as a single object the string composed of inner_path + cross_node
        paths = []
        for i in range(len(available_nodes)):
            for path in possible_paths[str(i)]:
                if path[-1] + label2 in available_lines:
                    paths.append(path + label2)
        return paths

    def propagate(self, signal_information):  # propagate signal info through specified path.
        first_node_label = signal_information.path[0]
        first_node = self.nodes[first_node_label]
        updated_s_information = first_node.propagate(signal_information)  # method from Node object
        return updated_s_information

    def draw(self):  # Name speaks for itself
        nodes_dict = self.nodes
        for node_label in nodes_dict:
            n0 = nodes_dict[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0, y0, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes_dict[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def weigh_paths(self, signal_power):
        df = DataFrame()  # DataFrame creation
        node_dict = self.nodes
        all_pairs = []
        all_paths = []  # All correct working paths in '->' notation
        latencies = []
        noise_powers = []
        snrs = []

        for label1 in node_dict:
            for label2 in node_dict:
                if label1 != label2:
                    all_pairs.append(label1 + label2)  # Generating every possible pair

        for pair in all_pairs:
            poss_paths = self.find_paths(pair[0], pair[1])  # All the possible paths that work for current pair
            for path in poss_paths:
                path_str = path_str2arrow(path)
                all_paths.append(path_str)
                # propagation per path
                signal_information = SignalInformation(signal_power, path)
                updated_information = self.propagate(signal_information)

                latencies.append(updated_information.latency)
                noise_powers.append(updated_information.noise_power)
                snrs.append(lin2db(updated_information.signal_power / updated_information.noise_power))

        df['Path'] = all_paths
        df['Latency'] = latencies
        df['Noise'] = noise_powers
        df['SNR'] = snrs
        df.set_index('Path', inplace=True)  # Performed changes to make path the actual index

        return df

    def find_best_snr(self, label1, label2):  # Returns best SNR path in string form and the channel
        best_path = None
        channel = -1
        paths = self.find_paths(label1, label2)  # Every possible admissible path (in str form without arrows)
        wp_df = self.weighted_paths
        highest_snr = float('-inf')

        modified_paths = [path_str2arrow(path) for path in paths]  # Convert to arrow paths

        for path in modified_paths:
            if wp_df.loc[path]['SNR'] > highest_snr:
                available_channel = self.path_available(path_arrow2str(path))  # Check availability of the path
                if available_channel != -1:
                    highest_snr = wp_df.loc[path]['SNR']
                    best_path = path_arrow2str(wp_df.loc[path].name)
                    channel = available_channel
        return best_path, channel

    def find_best_latency(self, label1, label2):  # Will return best path and channel used
        best_path = None
        channel = -1
        paths = self.find_paths(label1, label2)  # Every possible admissible path (in str form without arrows)
        wp_df = self.weighted_paths
        lowest_latency = float('inf')

        modified_paths = [path_str2arrow(path) for path in paths]  # Convert to arrow paths

        for path in modified_paths:
            if wp_df.loc[path]['Latency'] < lowest_latency:
                available_channel = self.path_available(path_arrow2str(path))
                if available_channel != -1:  # Check availability of the path
                    lowest_latency = wp_df.loc[path]['Latency']
                    best_path = path_arrow2str(wp_df.loc[path].name)
                    channel = available_channel
        return best_path, channel

    def stream(self, connection_list, choice='latency'):
        for connection in connection_list:
            if choice == 'snr':
                best_path_channel = self.find_best_snr(connection.input, connection.output)  # Returns a tuple
                if best_path_channel[0] is None:
                    connection.snr = 0.0
                    connection.latency = None
                    connection.bit_rate = 0.0
                    print(f'conn. {connection.input} to {connection.output} REJECTED (unavailable light paths)')
                else:
                    signal_information = LightPath(connection.signal_power, best_path_channel[0], best_path_channel[1])
                    first_node = self.nodes[best_path_channel[0][0]]  # (First node of path)
                    rb = self.calculate_bit_rate(best_path_channel[0], first_node.transceiver)  # Strategy
                    if rb != 0.0:
                        updated_info = self.propagate(signal_information)
                        connection.latency = updated_info.latency
                        connection.snr = 10 * (log10(updated_info.signal_power / updated_info.noise_power))
                        connection.bit_rate = rb
                        print('conn. accepted through path:', best_path_channel[0], 'channel:', best_path_channel[1])
                    else:
                        connection.latency = None
                        connection.snr = 0.0
                        print('conn. REJECTED through path:',
                              best_path_channel[0], 'channel:',
                              best_path_channel[1], '-> insufficient GSNR')
                        # Connection doesn't get deployed (lines and channels are kept free) if rb is 0
                        # rb 0 means connection didn't satisfy the gsnr threshold in the best AVAILABLE path

            elif choice == 'latency':
                best_path_channel = self.find_best_latency(connection.input, connection.output)
                if best_path_channel[0] is None:
                    connection.snr = 0.0
                    connection.latency = None
                    connection.bit_rate = 0.0
                    print(f'conn. {connection.input} to {connection.output} REJECTED due to unavailability')
                else:
                    signal_information = LightPath(connection.signal_power, best_path_channel[0], best_path_channel[1])
                    first_node = self.nodes[best_path_channel[0][0]]  # (First node of path)
                    rb = self.calculate_bit_rate(best_path_channel[0], first_node.transceiver)  # Strategy
                    if rb != 0.0:
                        updated_info = self.propagate(signal_information)
                        connection.latency = updated_info.latency
                        connection.snr = 10 * (log10(updated_info.signal_power / updated_info.noise_power))
                        connection.bit_rate = rb
                        print('conn. accepted through path:', best_path_channel[0], 'channel:', best_path_channel[1])
                    else:
                        print('conn. REJECTED through path:',
                              best_path_channel[0], 'channel:',
                              best_path_channel[1], '-> insufficient GSNR')

        self.update_route_space()

    def path_available(self, path, channel=0):  # Modified for receiving the channel number or -1 if no channel
        line_labels = []
        for i in range(len(path) - 1):
            line_labels.append(path[i] + path[i + 1])

        for i in range(channel, self.channels):  # If one specific channel is passed, check will start from that one on
            for line_label in line_labels:
                line = self.lines[line_label]
                if line.state[i] == 'occupied':
                    break
                if line_label == line_labels[-1]:
                    return i
        return -1

    def occupy_path(self, path, channel=0):  # Occupy every line of the path forcedly               (unused)
        line_labels = []
        for i in range(len(path) - 1):
            line_labels.append(path[i] + path[i + 1])
        for line_label in line_labels:
            line = self.lines[line_label]
            line.state[channel] = 'occupied'

    def free_path(self, path, channel=0):  # Free every line of the path forcedly                   (unused)
        line_labels = []
        for i in range(len(path) - 1):
            line_labels.append(path[i] + path[i + 1])
        for line_label in line_labels:
            line = self.lines[line_label]
            line.state[channel] = 'free'

    def update_route_space(self):
        all_paths = self.compute_all_paths()
        #  With all the possible paths, start the method
        channel_list_of_dicts = []
        index_list = []

        for path in all_paths:
            channel_dict = dict()

            for j in range(self.channels):
                channel_availability = 1  # Initialization of availability
                for i in range(len(path) - 1):
                    line_label = path[i] + path[i + 1]
                    in_node_label = path[i]
                    in_node = self.nodes[in_node_label]
                    out_node_label = path[i + 1]
                    current_state = self.lines[line_label].state[j]
                    if current_state == 'occupied':
                        current_state = 0
                    else:
                        current_state = 1
                    adj_matrix_entry = in_node.switching_matrix[in_node_label][out_node_label][j]
                    # AND operation (line state * s. matrix entry)
                    channel_availability *= adj_matrix_entry * current_state

                channel_dict[j] = int(channel_availability)  # After checking lines and corresponding switching m.

            index_list.append(path)
            channel_list_of_dicts.append(channel_dict)

        self.route_space = DataFrame(channel_list_of_dicts, index_list)

    def compute_all_paths(self):
        all_pairs = []
        all_paths = []
        node_dict = self.nodes
        for label1 in node_dict:
            for label2 in node_dict:
                if label1 != label2:
                    all_pairs.append(label1 + label2)  # Generating every possible pair
        for pair in all_pairs:
            possible_paths = self.find_paths(pair[0], pair[1])
            for path in possible_paths:
                all_paths.append(path)
        return all_paths

    def calculate_bit_rate(self, path, strategy):
        ber_t = 1e-3  # Bit error rate (assumed)
        rs = 32e9  # Symbol rate of lightpath in Hz (assumed)
        bn = 12.5e9  # Noise bandwidth in Hz (assumed)
        path_arrow = path_str2arrow(path)  # Use arrow notation to index the dataframe
        gsnr_db = self.weighted_paths.loc[path_arrow]['SNR']  # Take data from weighted paths
        gsnr = db2lin(gsnr_db)

        if strategy == 'fixed_rate':
            threshold = 2 * (erfcinv(2 * ber_t) ** 2) * (rs / bn)
            if gsnr >= threshold:
                rb = 100e9  # Bit rate in bps
            else:
                rb = 0

        elif strategy == 'flex_rate':
            threshold = 2 * (erfcinv(2 * ber_t) ** 2) * (rs / bn)
            threshold1 = (14 / 3) * (erfcinv((3 / 2) * ber_t) ** 2) * (rs / bn)
            threshold2 = 10 * (erfcinv((8 / 3) * ber_t) ** 2) * (rs / bn)
            if gsnr < threshold:
                rb = 0
            elif (gsnr >= threshold) and (gsnr < threshold1):
                rb = 100e9
            elif (gsnr >= threshold1) and (gsnr < threshold2):
                rb = 200e9
            else:
                rb = 400e9

        elif strategy == 'shannon':
            rb = (2 * rs * log2(1 + (gsnr * (bn / rs))))

        else:
            rb = 0

        return rb

    def traffic_matrix_deployment(self, traffic_matrix, blocking_event_limit):
        # Create dictionary for row, column correspondence to source, destination nodes

        number_node_dict = dict(enumerate(self.nodes.keys()))
        connection_list = []
        blocking_events = 0
        n_successful_connections = 0

        while self.check_finish(traffic_matrix) == 0 and blocking_events < blocking_event_limit:
            choice_entry = random.sample(list(range(len(traffic_matrix))), 2)  # pair of numbers inside the dimensions
            rb_requested = traffic_matrix[choice_entry[0]][choice_entry[1]]
            if rb_requested == 0.0:
                continue  # Sample entry of the matrix is either on the diagonal or the entry has already been taken
            node_in_label = number_node_dict[choice_entry[0]]
            node_out_label = number_node_dict[choice_entry[1]]

            connection = Connection(node_in_label, node_out_label, 1)
            self.stream([connection], 'snr')  # stream() method accepts list of connections
            if connection.bit_rate == 0.0:  # Connection couldn't be deployed
                blocking_events += 1
            else:
                n_successful_connections += 1
            subtraction = rb_requested - connection.bit_rate
            if subtraction < 0.0:  # Bit rate obtained normally is more than the one requested by matrix entry
                traffic_matrix[choice_entry[0]][choice_entry[1]] = 0.0
                connection.bit_rate = rb_requested  # Just deploy the capacity necessary to satisfy request
            else:
                traffic_matrix[choice_entry[0]][choice_entry[1]] = subtraction
            connection_list.append(connection)

        if blocking_events == blocking_event_limit:
            print(f'\nNetwork saturated: blocking event limit reached ({blocking_event_limit} events)')
        else:
            print('\nAll requests satisfied!')

        return connection_list, n_successful_connections

    @staticmethod
    def check_finish(traffic_matrix):
        for i in range(len(traffic_matrix)):
            for j in range(len(traffic_matrix[i])):
                if i != j:
                    if traffic_matrix[i][j] != 0.0:
                        return 0
        return 1


class Connection:
    def __init__(self, in_label, out_label, signal_power):
        self.input = in_label
        self.output = out_label
        self.signal_power = signal_power
        self.latency = 0.0
        self.snr = 0.0
        self.bit_rate = 0.0


class LightPath(SignalInformation):
    def __init__(self, s_power, path, channel=0):  # This call overrides SignalInformation class __init__
        super().__init__(s_power, path)  # Call __init__ from parent class SignalInformation
        self.channel = channel
        self.rs = 32e9
        self.df = 50e9

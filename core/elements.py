from pathlib import Path
import json
from math import sqrt
from scipy.constants import c
import matplotlib.pyplot as plt


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
        self.path[:] = self.path[1:]


class Node:
    def __init__(self, node_dict):
        self.label = node_dict['label']
        self.position = node_dict['position']
        self.connected_nodes = node_dict['connected_nodes']
        self.successive = dict()

    def propagate(self, signal_information):
        path = signal_information.path
        # if there are still elements to reach in the path list
        if len(signal_information.path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.path_update()  # delete path[0]
            signal_information = line.propagate(signal_information)
        return signal_information


class Line:
    def __init__(self, line_dict):
        self.label = line_dict['label']
        self.length = line_dict['length']
        self.successive = {}

    def latency_generation(self):
        # propagation velocity (2/3 c)
        latency = self.length / (2/3 * c)
        return latency

    def noise_generation(self, signal_power):
        noise = 1e-9 * signal_power * self.length
        return noise

    def propagate(self, signal_information):
        # add latency
        latency = self.latency_generation()  # generated the latency (proper for the line)
        signal_information.add_latency(latency)

        # add noise
        signal_power = signal_information.signal_power
        noise_power = self.noise_generation(signal_power)
        signal_information.add_noise_power(noise_power)

        # propagate (recursive)
        node_label = self.successive[signal_information.path[0]]
        signal_information = node_label.propagate(signal_information)
        return signal_information


class Network:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            json_nodes = json.load(f)
        self.nodes = dict()
        self.lines = dict()
        for node_key in json_nodes:
            # instantiate nodes
            node_dict = json_nodes[node_key]   # each json_node_key value is a dict (containing pos and connected nodes)
            node_dict['label'] = node_key      # manually add key-value pair because it's not in json file
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
                self.lines[line_label] = Line(line_dict)

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
            plt.plot(x0, y0, 'ro', markersize=10)
            plt.text(x0, y0, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes_dict[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

from pathlib import Path
import matplotlib.pyplot as plt
import random
from core.elements import Network
from core.elements import Connection


root = Path(__file__).parents[1]
json_path = root / 'Lab3' / 'nodes.json'

network = Network(json_path)
network.connect()
network.weighted_paths = network.weigh_paths(1)

node_labels = list(network.nodes.keys())
x_sample = []
latencies = []
connection_list = []

for i in range(20):  # Generate the random connections
    rand_pair = random.sample(node_labels, 2)
    connection_list.append(Connection(rand_pair[0], rand_pair[1], 1))
    x_sample.append(int(i))
    print(rand_pair)

network.stream(connection_list)  # You can also choose 'latency'
print('updated route space:\n', network.route_space)


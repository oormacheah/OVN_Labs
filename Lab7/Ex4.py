from pathlib import Path
import matplotlib.pyplot as plt
import random
from core.elements import Network
from core.elements import Connection


root = Path(__file__).parents[1]
json_path = root / 'Resources' / 'nodes_full_fixed_rate.json'

network = Network(json_path)
network.connect()
network.weighted_paths = network.weigh_paths(1)

node_labels = list(network.nodes.keys())
x_sample = []
connection_list = []
n_connections = 100


for i in range(n_connections):  # Generate the random connections
    rand_pair = random.sample(node_labels, 2)
    connection_list.append(Connection(rand_pair[0], rand_pair[1], 1))
    x_sample.append(int(i))
    print(rand_pair)

network.stream(connection_list, 'snr')  # You can also choose 'latency'
print('updated route space:\n', network.route_space)

bit_rates = []
tot_capacity = 0

for connection in connection_list:
    bit_rates.append(connection.bit_rate)
    tot_capacity += connection.bit_rate

avg_bit_rate = tot_capacity / len(connection_list)
print(f'Avg over {n_connections}:', '{:e}'.format(avg_bit_rate))
print('Total capacity used :', '{:e}'.format(tot_capacity))

# Plotting
plt.figure('aduaresgae')
plt.hist(bit_rates)
plt.xlabel('Rb')
plt.ylabel('Coincidences')
plt.grid(True)
plt.title('Bit Rate for i-th sample')
plt.show()

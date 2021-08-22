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

for i in range(100):  # Generate the random connections
    rand_pair = random.sample(node_labels, 2)
    connection_list.append(Connection(rand_pair[0], rand_pair[1], 1))
    x_sample.append(int(i))
    print(rand_pair)

network.stream(connection_list, 'snr')  # You can also choose 'latency'
print('updated route space:\n', network.route_space)

latencies = []
snrs = []

for connection in connection_list:
    latencies.append(connection.latency)
    snrs.append(connection.snr)

# Plotting
plt.figure()
plt.plot(x_sample, latencies)
plt.xlabel('i-th sample')
plt.ylabel('Latency')
plt.title('Latency')

plt.figure()
plt.plot(x_sample, snrs)
plt.xlabel('i-th sample')
plt.ylabel('SNR')
plt.title('SNR')

plt.show()

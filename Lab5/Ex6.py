from pathlib import Path
import matplotlib.pyplot as plt
import random
from core.elements import Network
from core.elements import Connection


root = Path(__file__).parents[1]
json_path = root / 'Resources' / 'nodes.json'

network = Network(json_path, 10)
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

network.stream(connection_list, 'snr')  # You can also choose 'latency'

latencies = []
snrs = []

for connection in connection_list:
    if connection.snr == 0.0:
        snrs.append(connection.snr)
        continue
    else:
        latencies.append(connection.latency)
        snrs.append(connection.snr)

# Plotting
plt.figure('Latency distribution')
plt.hist(latencies)
plt.xlabel('Latency')
plt.title('Latency distribution')

plt.figure('SNR distribution')
plt.hist(snrs)
plt.xlabel('SNR')
plt.title('SNR distribution')

plt.show()

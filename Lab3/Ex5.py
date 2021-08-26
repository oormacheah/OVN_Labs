from pathlib import Path
import pandas as pd
from core.conversions import lin2db
from core.elements import SignalInformation
from core.elements import Network


root = Path(__file__).parents[1]
json_path = root / 'Resources' / 'nodes.json'
network = Network(json_path)

df = pd.DataFrame()  # DataFrame creation
node_dict = network.nodes
line_dict = network.lines
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
    poss_paths = network.find_paths(pair[0], pair[1])  # All the possible paths that work for current pair
    for path in poss_paths:
        path_str = str()
        for node in path:
            path_str += node + '->'
        all_paths.append(path_str[:-2])
        # propagation per path
        signal_information = SignalInformation(1e-3, path)
        updated_information = network.propagate(signal_information)

        latencies.append(updated_information.latency)
        noise_powers.append(updated_information.noise_power)
        snrs.append(lin2db(updated_information.signal_power / updated_information.noise_power))

df['Path'] = all_paths
df['Latency'] = latencies
df['Noise'] = noise_powers
df['SNR'] = snrs

network.weighted_paths = df

print(df)


from pathlib import Path
import pandas as pd
from core.conversions import lin2db
from core.elements import SignalInformation
from core.elements import Network

root = Path(__file__).parents[1]
json_path = root / 'Lab3' / 'nodes.json'
network = Network(json_path)

best_path = network.find_best_snr2('A', 'F')
print(network.weighted_paths)
print(best_path)

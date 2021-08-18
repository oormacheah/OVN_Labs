from pathlib import Path
import pandas as pd
from core.conversions import lin2db
from core.conversions import db2lin
from core.elements import SignalInformation
from core.elements import Node
from core.elements import Line
from core.elements import Network


root = Path(__file__).parents[1]
json_path = root / 'Lab3' / 'nodes.json'
signal_info = SignalInformation(1e-3, 'ABD')
network = Network(json_path)
network.connect()
spectral_info = network.propagate(signal_info)
network.draw()


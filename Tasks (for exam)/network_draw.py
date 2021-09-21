from si_prefix import si_format
from pathlib import Path
import matplotlib.pyplot as plt
from core.elements import Network
from core.conversions import db2lin, lin2db


def main():
    root = Path(__file__).parents[1]
    json_path = root / 'Resources' / '262293.json'  # Insert the file to evaluate

    network = Network(json_path)
    network.connect()
    network.draw()


if __name__ == '__main__':
    main()

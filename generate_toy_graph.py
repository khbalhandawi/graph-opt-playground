"""Generate the synthetic toy benchmark graph (20 airports, 6 airlines).

Draws N_EDGES random (origin, destination, airline) records with a random
capacity each. Draws whose origin and destination coincide are discarded, so
with the default seed the graph keeps 1898 records on 375 of the 380 possible
directed segments. Repeated draws of the same airline on a segment are kept as
parallel edges; graph-opt pools their capacities when it builds the segment PMFs.

Writes ``data_toy/test_airline_nodes_E{N_EDGES}_V{N_NODES}_T{N_ATTRS}.graphml``
(and the matching airline ``.json``), the files read by ``solve_toy_benchmark.py``.
"""
import networkx as nx
import json
import os
import random


# Function to generate unique three-letter IATA codes
def generate_airport_iata_codes(num):
    iata_codes = []
    for i in range(26):
        for j in range(26):
            for k in range(26):
                code = chr(65 + i) + chr(65 + j) + chr(65 + k)
                iata_codes.append(code)
                if len(iata_codes) == num:
                    return iata_codes
    return iata_codes


# Generate unique two-letter IATA codes for the airlines
def generate_airline_iata_codes(num):
    iata_codes = []
    for i in range(26):
        for j in range(26):
            code = chr(65 + i) + chr(65 + j)
            iata_codes.append(code)
            if len(iata_codes) == num:
                return iata_codes
    return iata_codes


if __name__ == "__main__":

    random.seed(1234)

    N_EDGES = 2000
    N_NODES = 20
    N_ATTRS = 6

    # Create a MultiDiGraph
    MG = nx.MultiDiGraph()

    airport_iata_codes = generate_airport_iata_codes(N_NODES)
    # Add nodes with unique three-letter IATA codes
    for i, code in enumerate(airport_iata_codes):
        MG.add_node(str(i), iatacode=code,
                    longitude=-180 + 360 * (i / 19), latitude=-90 + 180 * (i / 19),
                    subregion=f"Region {(i % 5) + 1}", region="Global")

    airline_iata_codes = generate_airline_iata_codes(N_ATTRS)

    # Define the airlines with unique IATA codes
    airlines = {code: f'Airline {code}' for code in airline_iata_codes}

    # Add edges with airline IATA code
    for _ in range(N_EDGES):
        from_node = str(random.randint(0, N_NODES-1))
        to_node = str(random.randint(0, N_NODES-1))
        if from_node != to_node:
            airline_code = random.choice(list(airlines.keys()))
            MG.add_edge(from_node, to_node, edge_type=airline_code,
                        from_=MG.nodes[from_node]['iatacode'], to=MG.nodes[to_node]['iatacode'],
                        weight=random.randint(1000, 10000), market_share_percent=None)

    # Calculate total weights for each route
    total_weights = {}
    for (u, v, data) in MG.edges(data=True):
        if (u, v) not in total_weights:
            total_weights[(u, v)] = 0
        total_weights[(u, v)] += data['weight']

    # Set market share percentages
    for (u, v, data) in MG.edges(data=True):
        data['market_share_percent'] = (data['weight'] / total_weights[(u, v)]) * 100

    print(f"{MG.number_of_nodes()} airports, {MG.number_of_edges()} airline-segment records "
          f"on {len(total_weights)} directed segments")

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join(script_dir, "data_toy")
    os.makedirs(data_dir, exist_ok=True)
    stem = 'test_airline_nodes_E{}_V{}_T{}'.format(N_EDGES, N_NODES, N_ATTRS)
    nx.readwrite.graphml.write_graphml(MG, os.path.join(data_dir, stem + '.graphml'))  # store graph
    with open(os.path.join(data_dir, stem + '.json'), 'w') as file:
        json.dump(airlines, file, indent=4)

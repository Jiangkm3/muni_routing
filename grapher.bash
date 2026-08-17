#!/bin/bash
python plot_muni_route_named_streets.py "muni_route_streets.csv" --output muni_route_named_street_share.png &&
python plot_muni_route_named_streets.py "muni_directional_streets.csv" --output muni_route_named_directional_street_share.png &&
python plot_muni_route_named_streets_by_length.py "muni_route_streets.csv" --output muni_route_named_street_by_length_share.png &&
python plot_muni_route_named_streets_by_length.py "muni_directional_streets.csv" --output muni_route_named_directional_street_by_length_share.png
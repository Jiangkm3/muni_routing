#!/bin/bash
python plot_muni_route_named_streets.py "muni_route_streets.csv" --output muni_route_named_street_share.png --mapping-output muni_route_named_street_matches.csv &&
python plot_muni_route_named_streets.py "muni_directional_streets.csv" --output muni_route_named_directional_street_share.png --mapping-output muni_route_named_directional_street_matches.csv
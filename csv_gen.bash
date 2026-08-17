#!/bin/bash
python muni_route_street_analysis.py --aggregation scheduled --output muni_route_streets.csv &&
python muni_route_street_directional_analysis.py --aggregation scheduled --output muni_directional_streets.csv &&
python muni_route_stop_street_analysis.py --route-types 0,3,5 --output muni_stop_streets.csv && 
python muni_route_stop_street_analysis.py --route-types 0,3,5 --directional --output muni_stop_streets_directional.csv  
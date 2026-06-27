import os
import subprocess
import pandas as pd
import re

BINARY_PATH = '/content/booksim2/src/booksim'

def generate_config(routing, injection_rate, output_path):
    # Mapping internal names to BookSim2 valid routing function strings
    mapping = {'xy': 'dor', 'adaptive': 'min_adapt'}
    bs_routing = mapping.get(routing, routing)
    
    config_body = 'topology = mesh;\nk = 4;\nn = 2;\nrouting_function = {0};\ntraffic = uniform;\ninjection_rate = {1};\nwarmup_periods = 3;\nsample_period = 5000;\nmax_samples = 10;\nseed = 42;'.format(bs_routing, injection_rate)
    
    with open(output_path, 'w') as f:
        f.write('// Verified Q1 Config\n' + config_body)

def parse_log(log_path):
    if not os.path.exists(log_path): return None
    with open(log_path, 'r') as f:
        content = f.read()
        lat = re.search(r'Packet latency average\s*=\s*([\d\.]+)', content)
        thr = re.search(r'Throughput\s*=\s*([\d\.]+)', content)
        return {'latency': float(lat.group(1)) if lat else None, 'throughput': float(thr.group(1)) if thr else None}

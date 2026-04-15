import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence
import time


def _render(file_path, output_dir, num_views, file_name):
    output_folder = os.path.join(output_dir, file_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Evenly spaced yaw angles from 0 to 2π
    yaws = [2 * np.pi * i / num_views for i in range(num_views)]

    # Fixed pitch (0 means horizontal plane)
    pitchs = [0.0] * num_views

    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    args = [
        'python', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'BLENDER_EEVEE', #'CYCLES',BLENDER_EEVEE
        '--save_depth',
        '--save_mesh'
    ]
    
    with open('render.out', 'w') as out, open('render.err', 'w') as err:
        call(args, stdout=out, stderr=err)
    
    # call(args, stdout=DEVNULL, stderr=DEVNULL)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='Directory containing .glb files to render')
parser.add_argument('--output_dir', type=str, default='', help='Directory to save rendered images')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(input_dir):
    if file.endswith('.glb') and 'human' not in file:
        file_path = os.path.join(input_dir, file)
        print(f"Processing {file_path}")
        file_name = file.split('.')[0]
        _render(file_path, output_dir, 8, file_name)

        # pause for 5 seconds to avoid overloading the system
        print(f"Rendered {file_name} to {output_dir}")
    
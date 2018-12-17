# Compute average autonomous distance and success rate

import pickle
from numpy import array
from os import listdir

# Load data
src_path = 'PPO-performance/'
src_path = 'PPO_train-0.2_lane-performance/'
for file_name in listdir(src_path):
    with open(src_path + file_name, 'rb') as f: stats = pickle.load(f)

    # Numpy-fy
    distances = array(stats['distances'])
    arrived = array(stats['arrived'])

    # Convert to metres
    SCALE = 24 / 3.7  # px / m
    distances /= SCALE

    print(f'{file_name} - MAD: {distances.mean():.2f} m, arrival-rate: {arrived.mean()*100:.2f} %')

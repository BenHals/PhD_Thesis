import numpy as np
import pandas as pd
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
import skmultiflow.core
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..\GenerateDatastreamFiles\DriftStream\SyntheticWoodsmoke')
sys.path.append('..\DatasetGen\DriftStream\SyntheticWoodsmoke')
from windSimStream import WindSimGenerator
from windSimStream import PollutionSource
import pickle
import math
import matplotlib.animation as animation
from matplotlib.patches import Circle
from collections import deque

def do_update(stream, count, learner, r, w, counter):
    X, y = stream.next_sample()
    print(X[0].shape)
    # print(f"X: {X}, y: {y}")
    z = stream.get_last_image()

    count += 1

    p = learner.predict(X)
    is_correct = p == y
    if p[0] not in counter:
        counter[p[0]] = 0
    counter[p[0]] += 1
    if is_correct:
        r += 1
    else:
        w += 1
    learner.partial_fit(X,y)
    print(f"ACC: {r / (r+w)}, {counter}")

    return z, r, w, is_correct

def set_common(stream):
    stream.wind_direction = np.random.randint(0, 360)
    stream.wind_strength = ((np.random.rand() * 30) + 30) / (stream.window_width / 5)
    wind_direction_corrected = (stream.wind_direction - 90) % 360
    stream.wind_direction_radians = math.radians(wind_direction_corrected)

    stream.wind_strength_x = math.cos(stream.wind_direction_radians) * stream.wind_strength
    stream.wind_strength_y = math.sin(stream.wind_direction_radians) * stream.wind_strength

    stream.sources = []

    num_sources = 1
    for s in range(num_sources):
        x = (stream.window_width / 2) + stream.wind_strength_x * -50
        y = (stream.window_width / 2) + stream.wind_strength_y * -50
        strength = np.random.randint(10, 255)
        strength = 170
        size = np.random.randint(1, 4)
        stream.sources.append(PollutionSource(x, y, strength, (stream.window_width / 750) * size))
def set_ff(stream):
    stream.wind_direction = np.random.randint(0, 360)
    stream.wind_strength = ((np.random.rand() * 30) + 30) / (stream.window_width / 5)
    wind_direction_corrected = (stream.wind_direction - 90) % 360
    stream.wind_direction_radians = math.radians(wind_direction_corrected)

    stream.wind_strength_x = math.cos(stream.wind_direction_radians) * stream.wind_strength
    stream.wind_strength_y = math.sin(stream.wind_direction_radians) * stream.wind_strength

    stream.sources = []

    num_sources = 5
    for si, s in enumerate(range(num_sources)):
        r_w = 50 if si > 0 else 0
        x = (stream.window_width / 2 + random.randint(0, r_w) - r_w / 2) + stream.wind_strength_x * -40
        y = (stream.window_width / 2 + random.randint(0, r_w) - r_w / 2) + stream.wind_strength_y * -40
        strength = np.random.randint(10, 255)
        strength = 170
        size = np.random.randint(1, 4)
        stream.sources.append(PollutionSource(x, y, strength, (stream.window_width / 750) * size))

# fig, axs = plt.subplots(2, 2, sharey = 'row')
# for ci, concept in enumerate([(set_common, "Common"), (set_ff, "Rare")]):
    # stream = WindSimGenerator(sensor_pattern = "grid", num_sensors = 100, produce_image= True)

fig = plt.figure()
stream = WindSimGenerator(sensor_pattern = "grid", num_sensors = 100, produce_image= False, x_trail = 25)
set_ff(stream)
stream.prepare_for_use()
print(stream.ex)
learner = HoeffdingTree()





Writer = animation.writers['pillow']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

count = 0

r = 0
w = 0
counter = {}
learner = HoeffdingTree()


for ex in range(20000):
    z, r, w, c = do_update(stream, count, learner, r, w, counter)
    print(r / (r+w))

            
# def animate(i):
#     global count
#     global learner
#     global r
#     global w
#     global counter
#     global stream
#     z, r, w, c = do_update(stream, count, learner, r, w, counter)
#     print(r / (r+w))
#     plt.clf()
#     plt.imshow(z)

# ani = animation.FuncAnimation(fig, animate, init_func = lambda: [], repeat=True)
# plt.show()
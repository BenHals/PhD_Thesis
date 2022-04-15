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

def set_common(stream):
    stream.wind_direction = np.random.randint(0, 360)
    stream.wind_strength = ((np.random.rand() * 30) + 10) / (stream.window_width / 5)
    wind_direction_corrected = (stream.wind_direction - 90) % 360
    stream.wind_direction_radians = math.radians(wind_direction_corrected)

    stream.wind_strength_x = math.cos(stream.wind_direction_radians) * stream.wind_strength
    stream.wind_strength_y = math.sin(stream.wind_direction_radians) * stream.wind_strength

    stream.sources = []

    num_sources = 1
    for s in range(num_sources):
        x = (stream.window_width / 2) + stream.wind_strength_x * -75
        y = (stream.window_width / 2) + stream.wind_strength_y * -75
        strength = np.random.randint(10, 255)
        strength = 170
        size = np.random.randint(1, 4)
        stream.sources.append(PollutionSource(x, y, strength, (stream.window_width / 750) * size))
def set_ff(stream, sources = 25):
    stream.wind_direction = np.random.randint(0, 360)
    stream.wind_strength = ((np.random.rand() * 30) + 10) / (stream.window_width / 5)
    wind_direction_corrected = (stream.wind_direction - 90) % 360
    stream.wind_direction_radians = math.radians(wind_direction_corrected)

    stream.wind_strength_x = math.cos(stream.wind_direction_radians) * stream.wind_strength
    stream.wind_strength_y = math.sin(stream.wind_direction_radians) * stream.wind_strength

    stream.sources = []

    num_sources = sources
    for si,s in enumerate(range(num_sources)):
        r_w = 50 if si > 0 else 0
        x = (stream.window_width / 2 + random.randint(0, r_w) - r_w / 2) + stream.wind_strength_x * -75
        y = (stream.window_width / 2 + random.randint(0, r_w) - r_w / 2) + stream.wind_strength_y * -75
        strength = np.random.randint(10, 255)
        strength = 170
        size = np.random.randint(1, 4)
        stream.sources.append(PollutionSource(x, y, strength, (stream.window_width / 750) * size))

fig, axs = plt.subplots(2, 2, sharey = 'row', dpi=120)
ax2 = axs[1, 0].twinx()
ax3 = axs[1, 1].twinx()
secondary_axs = (ax2, ax3)
ax2.get_shared_y_axes().join(ax2, ax3)
# for ci, concept in enumerate([(set_common, "Common"), (set_ff, "Rare")]):
for ci, concept in enumerate([(set_ff, "Rare"), (set_common, "Common")]):
    # stream = WindSimGenerator(sensor_pattern = "grid", num_sensors = 100, produce_image= True)
    stream = WindSimGenerator(sensor_pattern = "grid", num_sensors = 100, produce_image= False)
    # stream = WindSimGenerator(sensor_pattern = "grid", num_sensors = 100, produce_image= False, x_trail = 25)
    concept[0](stream)
    stream.prepare_for_use()
    learner = HoeffdingTree()

    



    Writer = animation.writers['pillow']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

    count = 0

    r = 0
    w = 0
    reuse_right = 0
    reuse_wrong = 0
    counter = {}
    reuse_counter = {}
    sliding_window = deque()
    reuse_sliding_window = deque()


    reused = HoeffdingTree()
    

    # Pretrain
    reuse_len = 0
    reuse_acc = 0
    reuse_res = []
    pretrain_counts = {}
    image_frame = 10

    images = []
    # stream.produce_image = True
    for pex in range(40000):
        reuse_len += 1
        if pex == image_frame:
            stream.produce_image = True
        X,y = stream.next_sample()
        if pex == image_frame:
            images.append(stream.get_last_image())
            stream.produce_image = False
            image_frame = 4950
        # print(X)
        # print(y)
        p = reused.predict(X)
        if p[0] not in pretrain_counts:
            pretrain_counts[p[0]] = 0
        pretrain_counts[p[0]] += 1
        if p == y:
            reuse_acc += 1
        reused.partial_fit(X,y)
        if pex % 100 == 0:
            print(f"{pex}: {reuse_acc / reuse_len} {pretrain_counts} \r", end = "")
            reuse_res.append((reuse_len, reuse_acc / reuse_len))

            
    # fig1, ax1 = plt.subplots()
    # ax1.plot([x[0] for x in reuse_res], [x[1] for x in reuse_res])
    # plt.show()
    # print("")
    # print(f"")
    
    results = []
    num_train = 15000
    img_snapshop = random.randint(20, num_train)
    img_snapshop = 1

    retrained = HoeffdingTree()

    for ex in range(num_train):
        if ex == img_snapshop:
            stream.produce_image = True

        X,y = stream.next_sample()
        reuse_prediction = reused.predict(X)
        retrain_prediction = retrained.predict(X)


        # print(f"TEST: resue: {reuse_prediction}, retrain: {retrain_prediction}, y: {y}, {reuse_prediction == retrain_prediction}")
        if reuse_prediction[0] not in reuse_counter:
            reuse_counter[reuse_prediction[0]] = 0
        reuse_counter[reuse_prediction[0]] += 1

        if retrain_prediction[0] not in counter:
            counter[retrain_prediction[0]] = 0
        counter[retrain_prediction[0]] += 1

        reuse_iscorrect = reuse_prediction == y
        retrain_iscorrect = retrain_prediction == y

        reused.partial_fit(X, y)
        retrained.partial_fit(X, y)
        # print(reused.get_model_description())
        # print(retrained.get_model_description())
        if ex == img_snapshop:
            stream.produce_image = False
            z = stream.get_last_image()
            images.append(z)
        
        if reuse_iscorrect:
            reuse_right += 1
            reuse_sliding_window.append(1)
        else:
            reuse_wrong += 1
            reuse_sliding_window.append(0)
        if len(reuse_sliding_window) > 500:
            reuse_sliding_window.popleft()

        if retrain_iscorrect:
            r += 1
            sliding_window.append(1)
        else:
            w += 1
            sliding_window.append(0)
        if len(sliding_window) > 500:
            sliding_window.popleft()

        reuse_acc = sum(reuse_sliding_window) / len(reuse_sliding_window)
        retrain_acc = sum(sliding_window) / len(sliding_window)

        if ex % 250 == 0:
            print(f"{ex}: <{reuse_acc} {reuse_counter} | {retrain_acc} {counter}>")
        
        if len(sliding_window) > 50:
            results.append((ex, retrain_acc, reuse_acc, w, reuse_wrong))

    # for i in images:
    #     plt.imshow(i, cmap='gray', vmin = 0, vmax = 255)
    #     plt.show()
    print(f"Reuse wrong: {reuse_wrong}, Retrain wrong: {w}")
    with open(f"ff_res_{concept[1]}UN1.pickle", 'wb') as f:
        pickle.dump({'r': results, 'p': reused, 'l': retrained, 'z': z}, f)
    print([x[0] for x in results])
    print([x[1] for x in results])
    axs[1, ci].plot([x[0] for x in results], [x[1] for x in results], label = "retrain", color = 'tab:blue', ls = 'dashed')
    axs[1, ci].plot([x[0] for x in results], [x[2] for x in results], label = 'reuse', color = 'tab:blue')
    axs[1, ci].fill_between([x[0] for x in results], [x[1] for x in results], [x[2] for x in results], alpha = 0.3, color = 'tab:green', label = "Advantage")

    secondary_axs[ci].plot([x[0] for x in results], [x[3] for x in results], label = "retrain", color = 'tab:red', ls = 'dashed')
    secondary_axs[ci].plot([x[0] for x in results], [x[4] for x in results], label = 'reuse', color = 'tab:red')
    secondary_axs[ci].tick_params(axis='y', labelcolor='tab:red')
    axs[1, ci].tick_params(axis='y', labelcolor='tab:blue')
    # secondary_axs[ci].fill_between([x[0] for x in results], [x[3] for x in results], [x[4] for x in results], alpha = 0.3, color = 'green', label = "Advantage")
    
    axs[1, ci].set_xlabel("Observation")
    if ci == 0:
        axs[1, ci].set_ylabel("Recent Accuracy", color = 'tab:blue')
    if ci == 1:
        axs[1, ci].legend()
        secondary_axs[ci].set_ylabel('# Errors', color='tab:red')
    axs[0, ci].set_title(concept[1])
    axs[0, ci].imshow(z, cmap='gray', vmin = 0, vmax = 255)
plt.savefig(f"ff.png", dpi=200, bbox_inches='tight')
plt.savefig(f"ffH.png", dpi=500, bbox_inches='tight')
plt.savefig(f"ff.pdf", bbox_inches='tight')
plt.show()
# def animate(i):
#     global count
#     global learner
#     global r
#     global w
#     global counter
#     global stream
#     z, r, w = do_update(stream, count, learner, r, w, counter)
#     plt.clf()
#     plt.imshow(z)

# ani = animation.FuncAnimation(fig, animate, init_func = lambda: [], repeat=True)
# plt.show()
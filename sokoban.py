from pyswip import Prolog
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
import time
import pandas as pd
from PIL import Image
import cv2
import numpy as np

def get_swipl_version():
    prolog = Prolog()
    query = list(prolog.query("current_prolog_flag(version, Version)"))
    version_string = str(query[0]['Version'])
    major = int(version_string[:-4])
    minor = int(version_string[-4:-2])
    patch = int(version_string[-2:])
    print("Using swipl version {}.{}.{}".format(major, minor, patch))
    return major, minor, patch


swipl_major_version = get_swipl_version()[0]


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def map_moves(move):
    if move == "up":
        return 3
    elif move == "down":
        return 2
    elif move == "left":
        return 0
    elif move == "right":
        return 1


def scale(A, factor):
    return np.array(np.repeat(np.repeat(A, factor, 1), factor, 0), dtype=np.uint8)

def find_solution(size=8, num_boxes=2, time_limit=10, seed=0, verbose=1):
    dim_room = (size, size)

    env = SokobanEnvFast(dim_room=dim_room,
                         num_boxes=num_boxes,
                         seed=seed,
                         penalty_for_step=0)
    # The encoding of the board is described in README
    board = env.reset()

    wall = board[:,:,0] # this is a one-hot encoding of walls
    # For readibility first we deal with tops and then with rights
    tops = []
    for i in range(dim_room[0]):
        for j in range(dim_room[1]-1):
            if wall[i,j] == 0 and wall[i,j+1] == 0:
                tops.append("top(x{}y{},x{}y{})".format(i,j,i,j+1))

    rights = []
    for i in range(dim_room[0]-1):
        for j in range(dim_room[1]):
            if wall[i,j] == 0 and wall[i+1,j] == 0:
                rights.append("right(x{}y{},x{}y{})".format(i,j,i+1,j))

    boxes_initial_locations = board[:,:,4]
    boxes_initial = []
    for i in range(dim_room[0]):
        for j in range(dim_room[1]):
            if boxes_initial_locations[i,j] == 1:
                boxes_initial.append("box(x{}y{})".format(i,j))

    boxes_target_locations = board[:,:,2]
    boxes_target = []
    for i in range(dim_room[0]):
        for j in range(dim_room[1]):
            if boxes_target_locations[i,j] == 1:
                boxes_target.append("solution(x{}y{})".format(i,j))

    sokoban_initial_location = board[:,:,5] + board[:,:,6]
    for i in range(dim_room[0]):
        for j in range(dim_room[1]):
            if sokoban_initial_location[i,j] == 1:
                sokoban_string = "sokoban(x{}y{})".format(i,j)
                break

    tops_string = "[" + ','.join(tops) + ']'
    rights_string = "[" + ','.join(rights) + ']'
    boxes_initial_string = "[" + ','.join(boxes_initial) + ']'
    boxes_target_string = "[" + ','.join(boxes_target) + ']'

    prolog = Prolog()
    if swipl_major_version < 8:
        if verbose:
            print("Warning: using sokoban_swipl7.pl for compatibility with SWI-Prolog version 7")
        prolog.consult("sokoban_swipl7.pl")
    else:
        prolog.consult("sokoban.pl")

    query = "call_with_time_limit({},solve([{},{},{},{},{}],Solution))".format(time_limit,
                                                                               tops_string,
                                                                               rights_string,
                                                                               boxes_initial_string,
                                                                               boxes_target_string,
                                                                               sokoban_string)
    if swipl_major_version < 8:
        query = "use_module(library(time))," + query
    if verbose:
        print(query)
    try:
        result = list(prolog.query(query))
        rewards = []
        for i, r in enumerate(result):
            solution = r['Solution']
            actions = []
            frame = []
            for index in range(len(solution)):
                move = str(solution[index]).split()[-1]
                move = move[:-1]
                action = map_moves(move)
                actions.append(action)
                observation, reward, done, info = env.step(action)
                arr = scale(env.render(mode="rgb_array"), 4)
                frame.append(Image.fromarray(arr))
                rewards.append(reward)

            render_video(frame, f"{seed}_{i}_size{size}x{size}")

        if verbose:
            print("Last return {}".format(rewards[-1]))
        if rewards[-1] >= 10:
            return 1, actions
        else:
            return 0, []
    except:
        return 0, []


def render_video(frames, name):
    videodims = frames[0].size

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"videos/{name}.mp4", fourcc, 10, videodims)

    for f in frames:
        imtemp = f.copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()


if __name__ == "__main__":

    # initial_seed = 0    # 64
    # initial_seed = 100    # 61
    # initial_seed = 3333    # 52
    # initial_seed = 123456    # 70
    initial_seed = 666    # 59
    number_of_trials = 100
    time_start = time.time()

    df = pd.DataFrame(columns=['seed', 'actions'])

    results = 0
    not_my_fault = 0
    for seed in range(initial_seed, initial_seed + number_of_trials):
        print("Current trial {} result {}".format(seed, results))
        try:
            new_result, actions = find_solution(size=8, num_boxes=2, time_limit=20, seed=seed, verbose=0)
        except TypeError:
            not_my_fault += 1
            print(f"SEED {seed} produces WRONG board")
            new_result = 0
            actions = []

        results += new_result
        df = df.append({'seed': seed, 'actions': actions}, ignore_index=True)

    print("Number of solutions: {}".format(results))
    print("Number of fails that are not my fault: {}".format(not_my_fault))
    print("Total time: {}".format(time.time() - time_start))
    df.to_csv('results.csv')

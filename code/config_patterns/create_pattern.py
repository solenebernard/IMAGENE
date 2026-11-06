'''
Create the snake pattern of the smFISH tiles: for each position of the mosaic, the index of the tile (None where
there is no tile).

The pattern is saved in config_patterns/5x5.npy, the default layout of step 2 (fish.default_pattern in config.yaml).
An acquisition with another layout gets its own pattern in ROOTPATH/{EXP_PATH}/Fish/pattern_snake/{CONDITION}.npy,
which step 2 reads instead of the default one (tools/tools_fish.py).

Run from the folder code/: python3 -m config_patterns.create_pattern
'''


import numpy as np


def main():
    '''Create and save the pattern'''
    # Regular 5x5 snake pattern: the default one
    config_5x5 = np.array([
        [0, 1, 2, 3, 4],
        [9, 8, 7, 6, 5],
        [10, 11, 12, 13, 14],
        [19, 18, 17, 16, 15],
        [20, 21, 22, 23, 24]])
    np.save('./config_patterns/5x5.npy', config_5x5)


if __name__ == "__main__":
    main()

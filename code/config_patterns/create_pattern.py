import numpy as np

# Regular 5x5 snake pattern: the default one
config_5x5 = np.array([\
    [ 0,  1,  2,  3,  4], \
    [ 9,  8,  7,  6,  5], \
    [10, 11, 12, 13, 14], \
    [19, 18, 17, 16, 15], \
    [20, 21, 22, 23, 24]])
np.save('./config_patterns/5x5.npy', config_5x5)

# 4x6 pattern: 4 rows and 6 columns # VIH2 5-24H-cocktail
config_4x6 = np.array([\
    [ 0,  1,  2,  3,  4,  5], \
    [11, 10,  9,  8,  7,  6], \
    [12, 13, 14, 15, 16, 17], \
    [23, 22, 21, 20, 19, 18], \
    ])
np.save('./config_patterns/4x6.npy', config_4x6)
np.save(ROOTPATH + '2025-08-18_HIV2/Fish/pattern_snake/5-24H_cocktail.npy', config_4x6)

# 6x6 boots: # olga5 2-4H
config_6x6_boot1 = np.array([\
    [None, None, None,  0,  1,  2], \
    [None, None, None,  5,  4,  3], \
    [None, None, None,  6,  7,  8], \
    [None, None, None, 11, 10,  9], \
    [12,    13,   14,   15, 16, 17], \
    [23,    22,   21,   20, 19, 18]])
np.save('./config_patterns/config_6x6_boot1.npy', config_6x6_boot1)
np.save(ROOTPATH + '2025-09-21_Olga5/Fish/pattern_snake/2-4H.npy', config_6x6_boot1)

# 6x6 boots: # olga5 3-24H
config_6x5_boot1 = np.array([\
    [None, None,  0,  1,  2], \
    [None, None,  5,  4,  3], \
    [None, None,  6,  7,  8], \
    [13,   12,   11, 10,  9], \
    [14,   15,   16, 17, 18], \
    [23,   22,   21, 20, 19]])
np.save('./config_patterns/config_6x5_boot1.npy', config_6x5_boot1)
np.save(ROOTPATH + '2025-09-21_Olga5/Fish/pattern_snake/3-24H.npy', config_6x5_boot1)


# 6x4 pattern: 6 rows and 4 columns # olga5 4-MIX
config_6x4 = np.array([\
    [ 0,  1,  2,  3], \
    [ 7,  6,  5,  4], \
    [ 8,  9, 10, 11], \
    [15, 14, 13, 12], \
    [16, 17, 18, 19], \
    [23, 22, 21, 20], \
    ])
np.save('./config_patterns/6x4.npy', config_6x4)
np.save(ROOTPATH + '2025-09-21_Olga5/Fish/pattern_snake/4-MIX.npy', config_6x4)



feature_map_str = {
    'cat': [1, 0],
    'dog': [0, 1]
}

feature_map_int = [
    [1, 0, 0, 0, 0],
    # [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0],
]


def get_feature_by_num(num):
    return feature_map_int[num]

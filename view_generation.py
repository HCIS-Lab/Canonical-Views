import random
import time
import os

def generate_random_coordinates(N):
    """Generates N random (x, y, z) coordinates with a different random seed each time.

    Args:
        N (int): Number of random coordinates to generate.

    Returns:
        list: A list of tuples, each containing (x, y, z) coordinates.
    """
    # Use a different seed each time by combining time and os.urandom
    seed = int(time.time() * 1000000) ^ int.from_bytes(os.urandom(8), 'little')
    random.seed(seed)

    return [(random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)) for _ in range(N)]


def classify_angle(angle, offset):
    """
    Classifies an angle (0 to 360 degrees) into one of four categories:
    A: 0° ± 15°, B: 90° ± 15°, C: 180° ± 15°, D: 270° ± 15°.
    If it does not fall into any of these categories, it is classified as 'E'.

    Parameters:
    angle (float): The angle in degrees (0 to 360).

    Returns:
    str: Category ('A', 'B', 'C', 'D', or 'E').
    """

    categories = {
        'A': 0,
        'B': 90,
        'C': 180,
        'D': 270
    }

    for label, ref_angle in categories.items():
        lower_bound = (ref_angle - offset) % 360
        upper_bound = (ref_angle + offset) % 360
        
        # Handle cases where the range wraps around 0°
        if lower_bound > upper_bound:
            if angle >= lower_bound or angle <= upper_bound:
                return label
        else:
            if lower_bound <= angle <= upper_bound:
                return label

    return 'E'  # If no category is matched, return 'E'

offset = 15
views = generate_random_coordinates(500)

view_cat_num = {}
for t_0 in ['A','B','C','D', 'E']:
    for t_1 in ['A','B','C','D', 'E']:
        for t_2 in ['A','B','C','D', 'E']:
            view_cat_num[t_0+t_1+t_2] = 0
view_cat_num['non-planar'] = 0

view_dict = {}
cat_list = []
for i, view in enumerate(views):
    cat_0 = classify_angle(view[0], offset)
    cat_1 = classify_angle(view[1], offset)
    cat_2 = classify_angle(view[2], offset)

    if cat_0 != 'E' or cat_1 != 'E' or cat_2 != 'E':
        view_cat_num[cat_0+cat_1+cat_2] += 1
        cat_list.append(cat_0+cat_1+cat_2)
    else:
        view_cat_num['non-planar'] += 1
        cat_list.append('non-planar')




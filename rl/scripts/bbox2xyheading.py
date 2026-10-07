import math
import numpy as np

def get_bbox_traj_inverse(traj):
    res = []
    for bbox in traj:
        point = (bbox[0] + bbox[1])/2
        dx = (bbox[0][0] - bbox[1][0])
        dy = (bbox[0][1] - bbox[1][1])
        diagonal_angle = math.atan2(dy, dx)
        bbox_angle = math.atan2(2.0, 4.8)
        heading = diagonal_angle - bbox_angle
        res.append([point[0], point[1], heading])
    return np.array(res)
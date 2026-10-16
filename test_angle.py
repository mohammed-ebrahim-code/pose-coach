import numpy as np


a = np.array([2,2])
b = np.array([0,0])
c = np.array([3,0])
def calculate_angle(a,b,c):
    ba = a - b
    bc = c - b

    top = np.dot(ba , bc)
    ba_len = np.linalg.norm(ba)
    bc_len = np.linalg.norm(bc)
    val = np.clip(top / (ba_len * bc_len), -1 , 1)
    angle_radians = np.arccos(val)
    angle_dgree = np.degrees(angle_radians)
    print(angle_dgree)

calculate_angle(a,b,c)
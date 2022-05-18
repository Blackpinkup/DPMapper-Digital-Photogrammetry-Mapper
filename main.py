import os
import numpy as np
import cv2
from util import ply_registration



if __name__ == '__main__':
    mode = input('Point registration method(my / open3d): ')
    ply1_pth = '../data/DPEX_Data12/hand-high-tri.ply'
    ply2_pth = '../data/DPEX_Data12/hand-low-tri.ply'
    ply_registration(mode, ply1_pth, ply2_pth)
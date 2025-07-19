import copy
import json
import os
import time
import argparse
import random
import sys
import numpy as np
import torch
import math
from tqdm import tqdm
from typing import Dict, List, Optional
import trimesh
from scene import Scene
from se3dif.models.loader import load_model
from se3dif.samplers.grasp_samplers import Grasp_AnnealedLD
from se3dif.visualization import grasp_visualization
from visualize import visualize_point_cloud

p = np.load('/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/grasp_diffusion/se3dif/models/points/UniformPts.npy')
visualize_point_cloud(p, frame_size=0.1)

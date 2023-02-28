import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go


# import necessary packages
from typing import Sequence
import gradslam as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import TUM
from gradslam.slam import PointFusion
from torch.utils.data import DataLoader

path = "/scratch/saksham/datasets/tum"
poses_path = "inc_nerf_dumps/Nerf_0_phase1/saved_19"
sequences = "sequences.txt"

# load dataset
dataset = TUM(path, sequences = sequences , seqlen = 20, height = 480, width = 640, start = 56)
loader = DataLoader(dataset=dataset, batch_size=1)
colors, depths, intrinsics, poses, *_ = next(iter(loader))

# load other dataset
nerf_outputs = torch.load("renders/Nerf_0")
colors2 = nerf_outputs["rgb_pred"].unsqueeze(0)
depths2 = nerf_outputs["depth_pred"].unsqueeze(0).unsqueeze(-1)
poses2 = torch.load(poses_path)["poses"].unsqueeze(0)

# create rgbdimages object
rgbdimages = RGBDImages(colors2, depths2, intrinsics, poses2)

# step by step SLAM
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
slam = PointFusion(device=device)

pointclouds = Pointclouds(device=device)
pointclouds, recovered_poses = slam(rgbdimages)
fig = pointclouds.plotly(0, max_num_points=15000).update_layout(autosize=False, height = 1200, width=1200)

# Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig),
])

app.run_server(host='0.0.0.0', debug=True)
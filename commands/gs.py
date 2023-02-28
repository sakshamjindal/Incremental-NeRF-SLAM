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
sequences = "sequences.txt"

# load dataset
dataset = TUM(path, sequences = sequences , seqlen = 4, height = 60, width = 80)
loader = DataLoader(dataset=dataset, batch_size=32)
colors, depths, intrinsics, poses, *_ = next(iter(loader))

# create rgbdimages object
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)

# step by step SLAM
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
slam = PointFusion(device=device)

pointclouds = Pointclouds(device=device)
pointclouds, recovered_poses = slam(rgbdimages)
fig = pointclouds.plotly(0, max_num_points=15000).update_layout(autosize=False, width=600)

# Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig),
])

app.run_server(host='0.0.0.0', debug=True)
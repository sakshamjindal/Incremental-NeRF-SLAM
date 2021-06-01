from .blender import BlenderDataset
from .llff import LLFFDataset
from .blender_depth import BlenderDataset as BlenderDepth

dataset_dict = {'blender': BlenderDataset,
                'blender_depth': BlenderDepth,
                'llff': LLFFDataset}
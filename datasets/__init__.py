from .blender import BlenderDataset
from .llff import *
# from .llff_2 import *
# from .llff_tum import *
from .TUM import TUMDataset
from .llff_non_ndc import *

dataset_dict = {'blender': BlenderDataset,
                'llff': llff.LLFFDataset,
                'tum' : TUMDataset,
                'llff_non_ndc' : llff_non_ndc.LLFFDataset
                }
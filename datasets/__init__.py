from .blender import BlenderDataset
from .llff import *
# from .llff_2 import *
# from .llff_tum import *
from .TUM import TUMDataset
from .llff_non_ndc import *
from .llff_TUM import *
from .TUM_inv import *

dataset_dict = {'blender': BlenderDataset,
                'llff': llff.LLFFDataset,
                'tum' : TUMDataset,
                'tum_inv' : TUM_inv.TUMDataset,
                'llff_non_ndc' : llff_non_ndc.LLFFDataset,
                'llff_TUM' : llff_TUM.LLFFDataset
                }
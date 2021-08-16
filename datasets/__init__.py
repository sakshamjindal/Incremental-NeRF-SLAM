from .blender import BlenderDataset
from .llff import *
# from .llff_2 import *
# from .llff_tum import *
from .tum import TUMDataset
from .llff_non_ndc import *
# from .llff_TUM import *

dataset_dict = {'blender': BlenderDataset,
                'llff': llff.LLFFDataset,
                'tum' : TUMDataset,
                'llff_non_ndc' : llff_non_ndc.LLFFDataset,
                # 'llff_TUM' : llff_TUM.LLFFDataset
                }
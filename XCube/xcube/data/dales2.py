import os
import torch
import random
import math
from loguru import logger

from xcube.data.base import DatasetSpec as DS
from xcube.data.base import RandomSafeDataset

import fvdb
fvdb._Cpp.SparseGridBatch = fvdb._Cpp.GridBatch

import pickle

# to correctly load .pkl
custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler


class Dales2Dataset(RandomSafeDataset):
    def __init__(self, base_path, split, resolution, spec=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="scene", 
                 micro_key=[], voxel_num_interval=25000, car_voxel_num_interval=5000, 
                 duplicate_num=1, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution

        self.split = split
        if spec is None:
            self.spec = [DS.INPUT_PC]
        else:
            self.spec = spec
        
        # Get all items
        self.all_items = []
        split_file = os.path.join(base_path, (split + '.lst'))
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        if '' in models_c:
            models_c.remove('')
        self.all_items += [os.path.join(base_path, str(resolution), "%s.pkl" % m) for m in models_c]
        
        logger.info(f"Dales2Dataset: {len(self.all_items)} items")
        self.hparams = hparams
        
        # micro condition        
        self.micro_key = micro_key
        self.voxel_num_interval = voxel_num_interval
        self.car_voxel_num_interval = car_voxel_num_interval
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.all_items) * self.duplicate_num

    def get_name(self):
        return f"{self.custom_name}-{self.split}"
    
    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        data = {}
        input_data = torch.load(self.all_items[data_id % len(self.all_items)], pickle_module=custom_pickle)
        input_points = input_data['points']
        input_normals = input_data['normals'].jdata
        shape_name = self.all_items[data_id % len(self.all_items)]

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = shape_name

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = input_normals
        
        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = input_points
                
        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = input_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = input_normals
            
        if DS.GT_SEMANTIC in self.spec:
            data[DS.GT_SEMANTIC] = input_data["semantics"]
                
        if DS.LATENT_SEMANTIC in self.spec:
            latent_semantic = input_data["latent_semantics"]
            data[DS.LATENT_SEMANTIC] = latent_semantic
            
        if DS.MICRO in self.spec:
            micro = []
            H = (input_points.ijk.jdata[:, 0].max() -  input_points.ijk.jdata[:, 0].min()).item()
            W = (input_points.ijk.jdata[:, 1].max() -  input_points.ijk.jdata[:, 1].min()).item()
            D = (input_points.ijk.jdata[:, 2].max() -  input_points.ijk.jdata[:, 2].min()).item()

            H_ = 2 ** round(math.log2(H))
            W_ = 2 ** round(math.log2(W))
            D_ = 2 ** round(math.log2(D))
            N = 2 ** round(input_points.total_voxels / self.voxel_num_interval)
            
            C = (input_data["semantics"] == 2).sum().item() # car number
            C_ = 2 ** round(C / self.car_voxel_num_interval)
            
            R = 2 ** round((input_points.total_voxels - C) / self.voxel_num_interval)
            
            if "H" in self.micro_key:
                micro.append(H_)
            if "W" in self.micro_key:
                micro.append(W_)
            if "D" in self.micro_key:
                micro.append(D_)
            if "N" in self.micro_key:
                micro.append(N)
            if "C" in self.micro_key:
                micro.append(C_)
            if "R" in self.micro_key: 
                micro.append(R)
                
            data[DS.MICRO] = torch.Tensor(micro)

        return data

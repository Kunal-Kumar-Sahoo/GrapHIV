import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset

import deepchem as dc
from rdkit import Chem

import numpy as np
import pandas as pd
from tqdm import tqdm

import os

print(f"[INFO] Torch version: {torch.__version__}")
print(f"[INFO] CUDA availability: {torch.cuda.is_available()}")
print(f"[INFO] PyG version: {pyg.__version__}")


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root (str): The path to the directory where the dataset is stored.
        filename (str): The name of the `.csv` file inside the root directory.
        test (bool): Flag to check whether the dataset is training or testing dataset.
        transform: Transformations performed after loading the dataset
        pre_transform: Transformations performed before loading the dataset
        """
        super().__init__()
        self.test = test
        self.filename = filename
    
    @property
    def raw_file_names(self):
        """
        If this file exists in raw_dir, the download is not triggered
        """
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in `raw_dir`, data processing is skipped."""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f"data_test_{i}.pt" for i in list(self.data.index)]
        return [f"data_{i}.pt" for i in list(self.data.index)]
    
    def download(self):
        pass
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row["HIV_active"])
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f"data_test_{index}.pt"))
            else:
                torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))
    
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    def len(self):
        return self.data.shape[0]
    
    def get(self, index):
        """
        Equivalent to `__getitem__` in torch, and is not needed for PyG's `InMemoryDataset`
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f"data_test_{index}.pt"))
        else:
            data = torch.load(os.path.join(self.processed_dir, f"data_{index}.pt"))
        
        return data

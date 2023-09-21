from .hypersim import HypersimDataset
from .scannet_manhattan import ScanNetManhattanDataset
from .replica_semnerf import ReplicaSemNerfDataset


dataset_dict = {'hypersim': HypersimDataset,
                'scannet_manhattan': ScanNetManhattanDataset,
                'replica_semnerf': ReplicaSemNerfDataset}
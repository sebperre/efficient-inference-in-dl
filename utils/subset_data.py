import numpy as np
from torch.utils.data import Subset

def get_subset(dataset, subset_size=1000, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    return Subset(dataset, indices)

if __name__ == "__main__":
    print("This is a utils file and shouldn't be run directly.")
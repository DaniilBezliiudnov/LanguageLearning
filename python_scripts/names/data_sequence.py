import numpy as np
import data_preparer as data
from tensorflow.keras.utils import Sequence

class DataSequence(Sequence) :
    
    def __init__(self) -> None:
        self.x = data.training_data
        self.y = data.training_labels
        self.x_len = len(data.training_data)
        
    def __len__(self) -> int:
        return self.x_len
    
    def __getitem__(self, idx):
        return np.array(self.x), np.array(self.y)
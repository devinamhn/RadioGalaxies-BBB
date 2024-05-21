import torch
import torchvision.transforms as T

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from cata2data import CataData
from torch.utils.data import DataLoader
from mightee import MighteeZoo
from utils import Path_Handler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(150),  # Rescale to adjust for resolution difference between MIGHTEE & RGZ - was 70
        T.Normalize((1.59965605788234e-05,), (0.0038063037602458706,)),
    ]
)
paths = Path_Handler()._dict()
set = 'certain'

data = MighteeZoo(path=paths["mightee"], transform=transform, set="certain")
test_loader = DataLoader(data, batch_size=len(data))
for i, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

print(len(data))
print(len(y_test))

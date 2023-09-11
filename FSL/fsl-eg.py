from torchvision.datasets import Omniglot
from torchvision.transforms import ToTensor

dataset_path ='./'

transform = ToTensor()

trainset =  Omniglot(dataset_path, background=True, transform=transform, download=True)




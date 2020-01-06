import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pathlib
from PIL import Image 

__all__ = ['classificationDataSet']

class classificationDataSet(Dataset):
    """ a dataset composed of images which has to be used for classification

    Parameters
    ----------
    dataPath : str or Path or pathlib.Path to the directory containing the data
               it is assumed the data directory contains two sub-directories
               named img and labels that respectively contain images and 
               corresponding labels

    transform : torchvision.transforms 
                transformations to be applied to the images
                (default = None)

    Attributes:
    -----------
        imagePath : pathlib.path holding the path to the images
        labelPath : pathlib.path holding the path to the labels
        transform : torchvision.transforms, transformations that will be applied to the images



    """
    def __init__(self, dataPath, transform = None):            

        self.dataPath = pathlib.Path(dataPath)
        self.imagePath = dataPath / 'img' 
        self.labelPath = dataPath / 'labels'

        self.imagePath = [file for file in self.imagePath.iterdir() if file.is_file()]
        self.labels = [label for label in self.labelPath.iterdir() if label.is_file()]

        #TODO verify that there is a label for each img (name consistency) and handle
        #issues appropriately

        self.transform = transform

    def __getitem__(self, idx):
        # handle the image data
        image = Image.open(self.imagePath[idx])
        if self.transform:
            image = self.transform(image)
        # handle the label
        label = open(self.labels[idx]).readline().strip()
        # create the sample
        sample = {'image' : image, 'label' : label}
        # transform to Tensor
        transformer = transforms.ToTensor()
        sample['image'] = transformer(sample['image'])



        return sample 
    
    def __len__(self):
        return(len(self.imagePath))

from ..layers import layers
from torch import nn
from collections import OrderedDict
from torch.nn import ModuleDict

# Nice explanation about __all__
# http://xion.io/post/code/python-all-wild-imports.html
# __all__ is a convention and 
# - define the intended "public API" altough in Python, nothing is private in the end
# - defines what a wild import will actually import
__all__ = ['LeNet5']

class LeNet5(nn.Module):
    """ a simple Neural Network for classification which implements the LeNet-5 architecture

    Parameters:
    ----------
        input_channels (int, optional) : Number of input channels; default **1**

    
    Attributes:
    -----------
        layers : torch.nn.sequential holding the layers composing the conv2DAveragePool
                 i.e. a convolution, activation, average pooling and activation

    Methods:
    --------
        forward(x)
            performs the forward pass 



    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels

        # it's preferable to use torch ModuleDict or ModuleList to create layers
        # otherwhise it won't register correctly the child modules that are used
        # Give it a try ... use OrderedDict instead of ModuleDict and you shall 
        # see that it hinders the LeNet5.parameters() method ... It will return 
        # an empty list !
        self.layers = ModuleDict([('convLayers',nn.Sequential(
                                                layers.conv2DAveragePool(self.in_channels,6, 
                                                                        [5,2], [1,2], 0),
                                                layers.conv2DAveragePool(6, 16, 
                                                                        [5, 2], [1, 2], 0),
                                                nn.Conv2d(16, 120, 5,  1, 0),
                                                nn.BatchNorm2d(120),
                                                nn.ReLU()
                                                                )
                                    ) ,
                                    ('fcLayers',nn.Sequential(
                                                nn.Linear(120, 84),
                                                nn.BatchNorm1d(84),
                                                nn.ReLU(),
                                                nn.Linear(84,10)
                                                # the used loss function during
                                                # training requires raw numbers as inputs
                                                # no need for LogSoftmax as it will be applied
                                                # by the loss function
                                                # nn.LogSoftmax(dim=1)
                                                            )
                                    )
                                    ])

    def forward(self, x):
        out = self.layers['convLayers'](x)

        # important "trick"
        # when coming out of CNN, dimensions are [B, C, W, H]
        # while fully connected requires a 1D flattened Tensor
        # with dimensions [B, N] where N = CxWxH
        # a -1 dimension lets PyTorch compute this specific dimension
        # so command below says "whatever batch dimension" but 
        # a "definite number of rows based on CNN output"
        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])
        out = self.layers['fcLayers'](out)
        return out
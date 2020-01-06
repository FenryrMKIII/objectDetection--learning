from collections import Sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_all_ = ["conv2DAveragePool"]

class conv2DAveragePool(nn.Module):
    """ convenience layer that groups a 2D convolution (followed by activation) 
        and an average pooling (followed by activation)

    Parameters:
    ----------
        in_channels : int defining the number of input channels to the layer
        out_channels : int defining the number of output channels out of the layer
        kernel_size : int defining the kernel size 
        stride : int defining the stride
        padding : int defining the padding
        activation : torch.nn.modules.activation or 2-length sequence of 
                     nn.modules.activation, optional 
                     the activation functions for the conv2d & average
                     pooling layers. If a single str is provided, 
                     the same activation is applied to both layers. 
                     (default is nn.modules.activation.tanh)
    Attributes:
    -----------
        layers : torch.nn.sequential holding the layers composing the conv2DAveragePool
                 i.e. a convolution, activation, average pooling and activation
    methods:
    --------
        forward(x)
            performs the forward pass

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=nn.Tanh()):
        super().__init__()

        def parse_2Length_Sequence(x):
            """ checks if x is a 2-length sequence or a single item and 
                returns two separate items as appropriate

            Parameters:
            ----------
            """
            if isinstance(x, (Sequence, np.ndarray)):
                if len(x)==2:
                    item1 = x[0]
                    item2 = x[1]
                else:
                    raise ValueError('activation parameter has an unexpected'
                                    ' sequence length of ' + 
                                    len(x) + ' while one expects'
                                    'either a str or a 2-length sequence') #TODO is this correct way to catch such mistakes i.e. the if and the raise ? and the type of raise ?
                                                                            #TODO is this the correct kind of error message to send to the user ?
            else:
                item1 = item2 = x

            return item1, item2
        
        # Parameters
        self.in_channels = in_channels
        self.kernel_size = list(parse_2Length_Sequence(kernel_size)) #TODO if I have to check all arguments ... It this usually the case?? good practice ??
        self.out_channels = out_channels
        self.kernel_size = list(parse_2Length_Sequence(kernel_size))
        self.stride = list(parse_2Length_Sequence(stride))
        self.padding = list(parse_2Length_Sequence(padding))
        self.activation = list(parse_2Length_Sequence(activation))


        # Duck-typing priniciple in python : almost never check a 
        # variable type and be prepared to accept various types
        # e.g. a list, a tuple, ... if possible
        # duck-principle : if it resembles a duck treat it like a duck !

        self.layers = nn.Sequential(
            # One might remark that herebelow, the actual size of the input (usually, a picture i.e. a Tensor of size CxHxW)
            # is not defined ... only the channels (C) dimension is defined
            # looking into PyTorch documentation and code, one understands that 
            # nn.Conv2D indeed only requires these parameters.
            # BUT in nn.Conv2D code, nn.functional.conv2d gets called through the forward method
            # and this one (nn.functional.conv2d) requires input (BxCxHxW) to be defined (where B stands for the batch size)
            # so ... the call structure is not so straightforward when looking at PyTorch in details
            # one question remains ... How and when is forward method called ... ? Maybe there is a __call__ somewhere
            # that calls the forward method of each and every layer/module/... but I have not found this one yet
            # Looking further and going back-up the inheritance path, one finds that nn.Conv2D inherits nn.ConvN which
            # itself inherits module which has a __call__ method that calls self.forward
            # so ... there it is !
            nn.Conv2d(self.in_channels, self.out_channels, 
                      self.kernel_size[0], self.stride[0], 
                      self.padding[0], bias=True),
            nn.BatchNorm2d(out_channels), 
            self.activation[0],
            nn.MaxPool2d(self.kernel_size[1], 
                          self.stride[1], self.padding[1]),
            self.activation[1]
                                         )
    
    def forward(self, x):
        return self.layers(x)

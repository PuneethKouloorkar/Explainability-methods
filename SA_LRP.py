import cv2
import numpy 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
import copy
from matplotlib.colors import ListedColormap

# LRP implementation taken from https://git.tu-berlin.de/gmontavon/lrp-tutorial
# Pre-trained VGG-16 model is used here to classify ImageNet data 

# Load the test image
img = numpy.array(cv2.imread('castle.jpg'))[...,::-1]/255.0  # (224, 224, 3)

# Normalize the loaded test image as the training data was also normalised 
# with these values 
mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
X = (torch.FloatTensor(img[numpy.newaxis].transpose([0,3,1,2])*1) - mean) / std
X.requires_grad = True

# Import the pre-trained model and set it to evaluation mode
model = torchvision.models.vgg16(pretrained=True) 
model.eval()

# Visualizing data
def heatmap(R,sx,sy):
    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N)) # seismic colormap 
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.show()

# Clone a layer and pass its parameters through the function g
def newlayer(layer, g):
    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

# Convert VGG classifier's dense layers to convolutional layers
def toconv(layers):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer,nn.Linear):
            newlayer = None
            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers

# Load all the layers of the model. The classifer linear layers are converted 
# into equivalent 1x1 convolutions
layers = list(model._modules['features']) + \
         toconv(list(model._modules['classifier']))
L = len(layers)    # 38

# Propagate the input through the layers and record activations of all the layers
A = [X]+[None]*L
for l in range(L): 
    A[l+1] = layers[l].forward(A[l])
# The model classifies the image as 'castle' i.e. the neuron castle (index 483) has 
# the highest score. The relevance of the output layer is just the score of neuron 
# 483. zero-out all the other scores    

# Sensitivity Analysis-------------------------------------------------------------
A[-1][:,483,:,:].backward()
sa = A[0].grad
save_image(sa, f'SA.png')
# End of sensitivity Analysis------------------------------------------------------

# LRP------------------------------------------------------------------------------
T = torch.FloatTensor((1.0*(numpy.arange(1000)==483).reshape([1,1000,1,1])))
R = [None]*L + [(A[-1]*T).data]

# Propagate the network backwards, decomposing the prediction (or output relevance) 
# at every layer
# According to (14) section 10.3.2, max-pooling layers are treated as average pooling 
# layers in the backward pass
# Depending on the layer depth, various LRP rules are implemented
# The relevance is computed in 4 steps as described in (14)
for l in range(1,L)[::-1]:
    A[l] = (A[l].data).requires_grad_(True)       # To compute gradients w.r.t A[l]

    if isinstance(layers[l],torch.nn.MaxPool2d): 
        layers[l] = torch.nn.AvgPool2d(2)

    if isinstance(layers[l],torch.nn.Conv2d) or \
       isinstance(layers[l],torch.nn.AvgPool2d):

        if l <= 16:       
            rho = lambda p: p + 0.25*p.clamp(min=0)
            incr = lambda z: z+1e-9
        if 17 <= l <= 30: 
            rho = lambda p: p
            incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if l >= 31:       
            rho = lambda p: p
            incr = lambda z: z+1e-9

        z = incr(newlayer(layers[l],rho).forward(A[l]))               # step 1
        s = (R[l+1]/z).data                                           # step 2
        (z*s).sum().backward(); c = A[l].grad                         # step 3
        R[l] = (A[l]*c).data                                          # step 4
    else:
        R[l] = R[l+1]
# Note that the propagation is stopped one layer before the input
# One can also visualize the relevance at each layer

# LRP-zB rule for the input layer
A[0] = (A[0].data).requires_grad_(True)          # To compute gradients w.r.t A[0]

lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)          # step 1 (b)
z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)          # step 1 (c)
s = (R[1]/z).data                                                      # step 2
(z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

heatmap(numpy.array(R[0][0]).sum(axis=0),3.5,3.5)
# End of LRP------------------------------------------------------------------------
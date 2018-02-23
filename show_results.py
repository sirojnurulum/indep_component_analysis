import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def showimg(orig_images,mixed_images,component_images):
    d = len(orig_images)
        
    #use original images to correct for ambiguous sign of component images
    #first convert the images to arrays for forming dot product
    n1,n2  = orig_images[0].size
    n = n1*n2
    s = np.zeros((d,n))
    y = np.zeros((d,n))
    for k in range(d):
        sarr = np.asarray(orig_images[k])
        s[k,:] = np.reshape(sarr,(1,n)) - 127.0
        yarr = np.asarray(component_images[k])
        y[k,:] = np.reshape(yarr,(1,n)) - 127.0
    
    #find the sign of the component relative to the best match original
    ysgn = np.ones(d)
    for k in range(d):
        #find the original that corresponds to this component
        ys_ind = np.argmax(np.abs(np.dot(y[k,:],s.T)))
        #if the correlation is negative, set sign to -1 to correct 
        if np.dot(y[k,:],s[ys_ind,:]) < 0:
             ysgn[k] = -1
      
    #build list of component images with sign correction
    corrected_components = []
    for k in range(d):
        yarr = ysgn[k]*(np.asarray(component_images[k]) -127.0) + 127.0
        corrected_components.append(Image.fromarray(yarr))
        
             
    #set up grid of subplots, 3 x d, where each row is type of image
    #(original, mixed, component) and each column is one of the images in 
    #the list
    for sigtype in range(3):
        for ind in range(d):
            imgnum = sigtype*d +ind+1
            subplotnum = (300 + d*10 + imgnum)
            plt.subplot(subplotnum)
            plt.axis('off')
            if sigtype == 0:
                plt.title("original")
                plt.imshow(orig_images[ind])
            elif sigtype == 1:
                plt.title("mixed")
                plt.imshow(mixed_images[ind])        
            elif sigtype == 2:
                plt.title("component")
                plt.imshow(corrected_components[ind])  
 
    plt.show()


def showsig(orig_signals,mixed_signals,component_signals):
    #set up grid of subplots, 3 x d, where each row is type of signal
    #(original, mixed, component) and each column is one of the signals in 
    #the list
    d = len(orig_signals)
    for sigtype in range(3):
        for ind in range(d):
            imgnum = sigtype*d +ind+1
            subplotnum = (300 + d*10 + imgnum)
            plt.subplot(subplotnum)
            if sigtype == 0:
                plt.title("original")
                plt.plot(orig_signals[ind])
            elif sigtype == 1:
                plt.title("mixed")
                plt.plot(mixed_signals[ind])
            elif sigtype == 2:
                plt.title("component")
                plt.plot(component_signals[ind])
 
    plt.show()
        
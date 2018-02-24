import numpy as np
from PIL import Image
    
    
def createImg():
    """
    Create 3 images of the same size: load two different images of the same
    size from files, and create a third image of noise,
    and then mix the 3 images.
    The images are centered and normalized, so
    store the means and standard deviations of the images for display later.
    """
    d = 3;
    #for each image, convert to 1d array  for mixing
    #need to center to zero for mixing
    
    #first image from file
    prfimg = Image.open('profilepic2.jpg')
    prfimg = prfimg.convert('L')
    #convert to centered normalized 2d array
    prfarr = np.asarray(prfimg)
    prfarr = prfarr - 127.0
    #get image size
    n1,n2 = prfarr.shape
    n = n1*n2
    
    #second image from file
    astimg = Image.open('ashitaka.jpeg')
    astimg = astimg.convert('L')
    #convert to centered normalized 2d array
    astarr = np.asarray(astimg)
    astarr = astarr - 127.0
    n1tmp, n2tmp = astarr.shape
    
    #third image of noise, make a centered normalized arra
    noisarr = 120*np.sqrt(np.abs(np.random.randn(n1,n2)))
    noisarr = noisarr - 127.0
    
    #for storing 1-d arrays for mixing
    s = np.zeros((d,n))
    x = np.zeros((d,n))

    assert n1 == n1tmp
    assert n2 == n2tmp
        
    #reshape to 1-d arrays and store
    s[0,:] = np.reshape(prfarr,(1,n))
    s[1,:] = np.reshape(astarr,(1,n))
    s[2,:] = np.reshape(noisarr,(1,n))
    
    #mixing array
    a = np.array([[1, 0.4, 0.4], [-0.8, 0.8, 0.4],[0.7, -0.5, 0.8]])

    #mix signals
    x = np.dot(a,s)
    
    #create lists of original and mixed images
    orig_images = []
    mixed_images = []
    for k in range(3):
        sarr = np.reshape(s[k,:], (n1,n2)) + 127.0
        orig_images.append(Image.fromarray(sarr))
        xarr = np.reshape(x[k,:], (n1,n2)) + 127.0
        mixed_images.append(Image.fromarray(xarr))
    
    return (orig_images,mixed_images) 
    
    
def createSig():
    """
    create 2 1-d signals: one a modulated sinusoid, the other filtered noise,
    and mix them
    """
    n=1000
    s = np.zeros((2,n))
    t = np.arange(n)
    s[0,:] = np.sin(2*np.pi*t/50)*np.sin(2*np.pi*t/500);
    h = 1.5*np.convolve(np.ones(30)/30.0,np.ones(20)/20,mode='full')
    s[1,:] = np.convolve(np.random.randn(n),h,mode='same')
    
    #mixing array
    a = np.array([[1, 0.5], [1.75, -2]])
    x = np.dot(a,s)
    
    #create lists of original and mixed signals (1d numpy arrays)
    orig_signals = []
    mixed_signals = []
    for k in range(2):
        orig_signals.append(s[k,:])
        mixed_signals.append(x[k,:])
        
    return (orig_signals, mixed_signals)    
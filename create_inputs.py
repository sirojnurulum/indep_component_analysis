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
    mn=np.zeros(d)
    sd=np.zeros(d)
    
    #first image from file
    prpic = Image.open('profilepic2.jpg')
    prpic = prpic.convert('L')

    prpic = np.asarray(prpic)
    mn[0] = np.mean(prpic)
    sd[0] = np.std(prpic)
    prpic = (prpic - mn[0])/sd[0];
    n1,n2 = prpic.shape
    n = n1*n2
    
    #second image from file
    aspic = Image.open('ashitaka.jpeg')
    aspic = aspic.convert('L')

    aspic = np.asarray(aspic)
    mn[1] = np.mean(aspic)
    sd[1] = np.std(aspic)
    aspic = (aspic - mn[1])/sd[1];
    n1tmp, n2tmp = aspic.shape
    
    #third image of noise
    noispic = 120*np.sqrt(np.abs(np.random.randn(n1,n2)))
    mn[2] = np.mean(noispic)
    sd[2] = np.std(noispic)
    noispic = (noispic - mn[2])/sd[2];
    
    
    s = np.zeros((d,n))
    x = np.zeros((d,n))

    assert n1 == n1tmp
    assert n2 == n2tmp
        
    #reshape to 1-d arrays and store
    s[0,:] = np.reshape(prpic,(1,n))
    s[1,:] = np.reshape(aspic,(1,n))
    s[2,:] = np.reshape(noispic,(1,n))
    
    #mixing array
    a = np.array([[1, 0.75, 0.6], [0.75, 1, -0.5],[0.7, 0.6, 1]])

    x = np.dot(a,s)
    return (s,x,n1,n2,mn,sd) 
    
    
def createSig():
    """
    create 2 1-d signals: one a modulated sinusoid, the other filtered noise,
    and mix them
    """
    n=100
    s = np.zeros((2,n))
    t = np.arange(n)
    s[0,:] = np.sin(2*np.pi*t/3)*np.sin(2*np.pi*t/30);
    h = np.ones(3)/3.0
    s[1,:] = np.convolve(np.random.randn(n),h,mode='same')
    
    #mixing array
    a = np.array([[1, 0.5], [1.75, -2]])
    x = np.dot(a,s)
    return (s,x)    
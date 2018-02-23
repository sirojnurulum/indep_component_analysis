import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def showimg(s,x,y,n1,n2,mn,sd):
    d,n = s.shape
    for sigtype in range(3):
        for ind in range(d):
            imgnum = sigtype*d +ind+1
            subplotnum = (300 + d*10 + imgnum)
            plt.subplot(subplotnum)
            plt.axis('off')
            if sigtype == 0:
                spic = Image.fromarray(np.reshape(s[ind,:]*sd[ind]+mn[ind],(n1,n2)))
                plt.imshow(spic)
            elif sigtype == 1:
                xpic = Image.fromarray(np.reshape(x[ind,:]*sd[ind]+mn[ind],(n1,n2)))
                plt.imshow(xpic)        
            elif sigtype == 2:
                #find the s that this y corresponds to
                ys_ind = np.argmax(np.abs(np.dot(y[ind,:],s.T)))
                ysgn = 1
                if np.dot(y[ind,:],s[ys_ind,:]) < 0:
                    ysgn = -1
                ypic = Image.fromarray(np.reshape(ysgn*y[ind,:]*sd[ys_ind]+mn[ys_ind],(n1,n2)))
                plt.imshow(ypic)  
 
    plt.show()

def showsig(s,x,y):
    d,n = s.shape
    for sigtype in range(3):
        for ind in range(d):
            imgnum = sigtype*d +ind+1
            subplotnum = (300 + d*10 + imgnum)
            plt.subplot(subplotnum)
            if sigtype == 0:
                plt.plot(s[ind,:])
            elif sigtype == 1:
                plt.plot(x[ind,:])
            elif sigtype == 2:
                plt.plot(y[ind,:])
 
    plt.show()
        
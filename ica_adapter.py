import numpy as np
from PIL import Image
import ica_algorithm

class ICA_Adapter:
    ica_alg = ica_algorithm.ICA_Algorithm()
    
     
class ICA_Image(ICA_Adapter):
    def runICA(self,mixed_images):
        d = len(mixed_images)
        n1,n2  = mixed_images[0].size
        n = n1*n2
        
        #put the mixed images in a form for ica
        x = np.zeros((d,n))
        for k in range(d):
            xarr = np.asarray(mixed_images[k])
            x[k,:] = np.reshape(xarr,(1,n))
            
        #call ica
        y = self.ica_alg.ica(x)
        
        #create output list of component
        component_images = []
        for k in range(d):
            #ica centers and normalizes, so
            #find out how to scale the output, assuming it has zero mean
            yarr = np.reshape(y[k,:],(n1,n2))*64.0 + 127.0
            component_images.append(Image.fromarray(yarr))
    
        return component_images

class ICA_Signal(ICA_Adapter):
    def runICA(self,mixed_signals):
         d = len(mixed_signals)
         n = len(mixed_signals[0])
         
         #crate array to hold all 1-d signals
         x = np.zeros((d,n))
         for k in range(d):
             x[k,:] = mixed_signals[k]
             
         #call ica
         y = self.ica_alg.ica(x)
         
         #put component signals back into a list
         component_signals =  []
         for k in range(d):
             component_signals.append(y[k,:])
             
         return component_signals
         
         
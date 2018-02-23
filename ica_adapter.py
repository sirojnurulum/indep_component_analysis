import numpy as np
from PIL import Image

class ICA_Adapter:
    def ica(self,x):
        d,n = x.shape
        maxiter = 20
        
        #center and normalize the inputs
        for k in range(d):
            mn = np.mean(x[k,:])
            sd = np.std(x[k,:])
            x[k,:] = (x[k,:] - mn)/sd
        
        #whiten the inputs
        c = np.cov(x)
        D,E = np.linalg.eig(c)
        D = np.diag(np.power(D,-0.5))
        x = np.dot(np.dot(np.dot(E,D),E.conj().T),x)
        
        #set up weight vectors per component
        wMat = np.zeros((d,d))
        
        #for each component:
        for p in range(d):
            #initialize iteration loop for this component
            dotw = 100.0
            iter = 0
            w = np.random.randn(d)
            w = w / np.linalg.norm(w,2)
            
            while (abs(dotw-1.0) > 0.0001) and (iter < maxiter):
                #fast ica algorithm
                z = np.dot(w,x)
                g = np.tanh(z)
                gp = 1.0-np.square(g)
                Exg = np.zeros(d)
                Egp = 0;
                for i in range(n):
                    Exg = Exg + x[:,i]*g[i]
                    Egp = Egp + gp[i]
                Exg = Exg/n
                Egp = Egp/n
                wold = w
                w = Exg - Egp*w                
                
                #if not the first component, make it the weight vector
                #orthogonal to the previous ones
                if p > 0:
                    sumproj = np.zeros(d)
                    for k in range(p):
                        sumproj = sumproj + np.dot(w,wMat[k,:])*wMat[k,:]
                    w = w - sumproj
                #normalize weights
                w = w/np.linalg.norm(w,2);    
                
                #update change in weights for stopping criteria
                dotw = abs(np.dot(w,wold));
                iter=iter+1;
                print("iter= ",iter,", dotw = ",dotw)  
                
            wMat[p,:]= w;
            
        y = np.dot(wMat, x);    
        
        return y
        
        
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
        y = super(ICA_Image,self).ica(x)
        
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
         y = super(ICA_Signal,self).ica(x)
         
         #put component signals back into a list
         component_signals =  []
         for k in range(d):
             component_signals.append(y[k,:])
             
         return component_signals
         
         
import numpy as np

class ICA_Algorithm:
    """
    Container class for the method y=ica(x).
    The input x is a d x n array of float, corresponding 
    to d signals or images, each of which has n samples.
    The output y is a d x n array of float, corresponding 
    to d signal or images components, from which the inputs were made.
    Each of the signal in y has been centered to zero mean and unit standard
    deviation.
    """
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
        
 

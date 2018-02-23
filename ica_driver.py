import create_inputs
import ica_adapter
import show_results

#which test case to run
flag = 1

if flag  == 0:
    s,x,n1,n2,mn,sd = create_inputs.createImg()
    icaimg = ica_adapter.ICA_Image();
    y = icaimg.runICA(x)
    show_results.showimg(s,x,y,n1,n2,mn,sd)
else:
    s,x = create_inputs.createSig()
    icasig = ica_adapter.ICA_Signal()
    y = icasig.runICA(x)
    show_results.showsig(s,x,y)


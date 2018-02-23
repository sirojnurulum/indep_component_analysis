import create_inputs
import ica_adapter
import show_results

#which test case to run
flag = 0

if flag  == 0:
    orig_images,mixed_images = create_inputs.createImg()
    icaimg = ica_adapter.ICA_Image();
    component_images = icaimg.runICA(mixed_images)
    show_results.showimg(orig_images,mixed_images,component_images)
else:
    orig_signals,mixed_signals = create_inputs.createSig()
    icasig = ica_adapter.ICA_Signal()
    component_signals = icasig.runICA(mixed_signals)
    show_results.showsig(orig_signals,mixed_signals,component_signals)


import numpy as np

imagefile1 = ('/Users/haohuiche/Desktop/WUSTL/FL24/PLab/pyabel_new/PyAbel-master/apply'
              '/supplementary_imageGen_burner_tests/first add then trnasform then add then transform then add '
              'noise.dat')
IM1 = np.loadtxt(imagefile1)
imagefile2 = ('/Users/haohuiche/Desktop/WUSTL/FL24/PLab/pyabel_new/PyAbel-master/apply'
              '/supplementary_imageGen_burner_tests/first add peaks then transform then add noise.dat')
IM2 = np.loadtxt(imagefile1)

error = np.abs(IM1 - IM2)
mse = np.mean(error**2)
print(f"Mean Squared Error (MSE): {mse:.6f}")
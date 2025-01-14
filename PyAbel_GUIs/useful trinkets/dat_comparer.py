import numpy as np

imagefile1 = ('your file1 path')
IM1 = np.loadtxt(imagefile1)
imagefile2 = ('your file2 path')
IM2 = np.loadtxt(imagefile1)

error = np.abs(IM1 - IM2)
mse = np.mean(error**2)
print(f"Mean Squared Error (MSE): {mse:.6f}")
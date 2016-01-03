%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('./Desktop/DR_Kaggle/sample/13_left.jpeg')
plt.imshow(img)
plt.show()



import skimage.color as skcolor
img_gray = skcolor.rgb2gray(img[:,:,1])
plt.imshow(img_gray,cmap = plt.cm.gray)
#plt.show()



from skimage.morphology import erosion,dilation
from skimage.morphology import disk
import numpy as np

selem1 = disk(9.21)
selem2 = disk(7.21)

dilated1 = dilation(img_gray, selem1)
dilated2 = dilation(img_gray, selem2)

dilated = np.subtract(dilated1, dilated2)

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(dilated1, cmap=plt.cm.gray)
#ax[1].imshow(dilated2, cmap=plt.cm.gray)

plt.imshow(dilated,cmap=plt.cm.gray)




import skimage
# Get the histogram data
hist_phase, bins_phase = skimage.exposure.histogram(dilated)

# Use matplotlib to make a pretty plot of histogram data
plt.fill_between(bins_phase, hist_phase, alpha=0.5, color='b')

# Label axes
plt.xlabel('pixel value')
plt.ylabel('count')




# Threshold value, as obtained by eye
thresh_phase1 = 40
# Generate thresholded image
im_phase_1 = dilated > thresh_phase1
binar_1 = 255 - im_phase_1

print im_phase_1.max()
# Display phase and thresholded image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(dilated, cmap=plt.cm.gray)
ax[1].imshow(im_phase_1, cmap=plt.cm.gray)

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(subtracted1, cmap=plt.cm.gray)
#ax[1].imshow(im_phase_1, cmap=plt.cm.gray)




import numpy as np
from skimage.morphology import erosion,dilation, reconstruction
from skimage.morphology import diamond,disk

#h= 0.1
seed = np.copy(im_phase_1)
seed[1:-1,1:-1] = im_phase_1.min()

mask = im_phase_1
filled = reconstruction(seed,mask,method='dilation',selem=diamond(10))
contour = im_phase_1 - filled
plt.imshow(contour, cmap=plt.cm.gray)



selem0 = disk(1)
selem1 = diamond(1)
selem2 = diamond(1)

dilated2 = dilation(contour, selem0)
erosed2 = erosion(dilated2, selem1)
dilated3 = dilation(erosed2, selem1)
erosed3 = erosion(dilated3, selem2)

fig, ax = plt.subplots(1, 4, figsize=(10, 5))
ax[0].imshow(dilated2, cmap=plt.cm.gray)
ax[1].imshow(erosed2, cmap=plt.cm.gray)
ax[2].imshow(dilated3, cmap=plt.cm.gray)
ax[3].imshow(erosed3, cmap=plt.cm.gray)



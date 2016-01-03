%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('./Desktop/DR_Kaggle/sample/13_right.jpeg')
plt.imshow(img)
plt.show()
print img



import skimage.color as skcolor
img_gray = skcolor.rgb2gray(img[:,:,1])
plt.imshow(img_gray,cmap = plt.cm.gray)
#plt.show()


from skimage import exposure 
im2 = exposure.equalize_adapthist(img_gray,clip_limit=0.08)
plt.imshow(im2, cmap=plt.cm.gray)
plt.show
print im2



from skimage.morphology import erosion,dilation
from skimage.morphology import disk
selem1 = disk(5.21)
dilated1 = dilation(im2, selem1)
closing1 = erosion(dilated1,selem1)

plt.imshow(closing1, cmap = plt.cm.gray)



selem2 = disk(3.472)
dilated2 = dilation(im2, selem2)
closing2 = erosion(dilated2,selem2)
closing_2= 255 -closing2
plt.imshow(closing2, cmap = plt.cm.gray)




selem3 = disk(0.7)
dilated3 = dilation(im2, selem3)
closing3 = erosion(dilated3,selem3)
closing_3=255-closing3
print closing3
print closing3.shape
plt.imshow(closing3, cmap = plt.cm.gray)





import numpy as np
subtracted1 = np.subtract(closing1,closing2)
subtracted2 = np.subtract(closing1 ,closing3)
print subtracted1

fig, (ax0, ax1) = plt.subplots(ncols =2)
ax0.imshow(subtracted1, cmap = plt.cm.gray)
ax1.imshow(subtracted2, cmap = plt.cm.gray)
plt.tight_layout





import skimage
# Get the histogram data
hist_phase, bins_phase = skimage.exposure.histogram(subtracted2)
hist_phase1, bins_phase1 = skimage.exposure.histogram(subtracted1)
# Use matplotlib to make a pretty plot of histogram data
#plt.fill_between(bins_phase, hist_phase, alpha=0.5, color='b')
plt.fill_between(bins_phase1, hist_phase1, alpha=0.5, color='b')
# Label axes
plt.xlabel('pixel value')
plt.ylabel('count')





# Use matplotlib to make a pretty plot of histogram data
plt.fill_between(bins_phase, hist_phase, alpha=0.5, color='b')
plt.plot([0.05, 0.05], [0, 70000], 'r-')

# Label axes
plt.xlabel('pixel value')
plt.ylabel('count')





# Threshold value, as obtained by eye
thresh_phase1 = 0.4
thresh_phase = 0.2
# Generate thresholded image
im_phase_2 = subtracted2 > thresh_phase1

im_phase_1 = subtracted1 > thresh_phase

# Display phase and thresholded image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(subtracted2, cmap=plt.cm.gray)
ax[1].imshow(im_phase_2, cmap=plt.cm.gray)

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(subtracted1, cmap=plt.cm.gray)
#ax[1].imshow(im_phase_1, cmap=plt.cm.gray)






from skimage.morphology import disk
from skimage.filters.rank import median
med2 = median(im_phase_2, disk(0.7))
inv_med2 = np.invert(med2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[1].imshow(med2, cmap=plt.cm.gray)
#ax[0].imshow(im_phase_2, cmap=plt.cm.gray)
ax[0].imshow(inv_med2, cmap=plt.cm.gray)
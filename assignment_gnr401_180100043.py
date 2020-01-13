import numpy as np
import pylab as plt
import matplotlib.image as mpimg
import scipy.io as sio
from scipy.io import loadmat
from PIL import Image, ImageFilter
from scipy import ndimage as ndi


def imhist(im):
	#creates histogram
	m,n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cum_sum(h):
	#cumulative sum for equalization
	return [sum(h[:i+1]) for i in range(len(h))]

def hist_eq(im):
	#create histogram from image
	h = imhist(im)

	#cumulative distribution
	cdf = np.array(cum_sum(h)) 

	#transfer function
	sk = np.uint8(255 * cdf) 
	s1, s2 = im.shape
	Y = np.zeros_like(im)

	# transferred values
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)

	#returns new image, original and new histogram 
	return Y , h, H

def edge_detection(img):
	#sobel implementation for edge detection
    edgy = np.copy(img)
    size = edgy.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            edgy[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return edgy

#data loading
data = sio.loadmat('Indian_pines_gt.mat')
data = data['indian_pines_gt']
img = Image.fromarray(data)
img.save('test_image.png')

#image to 2D array
img = np.array(img)

#adding gaussian noise
mean = 0
var = 0.1
sd = var**0.5
gauss = np.random.normal(mean,sd,img.shape)
gauss = gauss.reshape(img.shape)
gauss = gauss.astype(int)
noisy = img + gauss


new_img, h, new_h = hist_eq(noisy)

# show old and new image
# show original image
plt.subplot(121)
plt.imshow(noisy)
plt.title('original image with noise')
plt.set_cmap('gray')

# show equalized image
plt.subplot(122)
plt.imshow(new_img)
plt.title('histogram equalized image')
plt.set_cmap('gray')

plt.show()

# plot histograms 
# original histogram
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram') 

#histogram of equalized image
fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram') 

plt.show()

#generate edgy image
edgy = edge_detection(img)

#add edgy image to original to generate sharpened image
sharpened = img + edgy

#plot edgy and sharpened images
plt.subplot(121)
plt.imshow(edgy)
plt.title('image edges')
plt.set_cmap('gray')

plt.subplot(122)
plt.imshow(sharpened)
plt.title('sharpened image')
plt.set_cmap('gray')

plt.show()
from PIL import Image
import numpy as np

def main():
    imgfilename = 'images/papatya.jpeg'
    img = readPILimg(imgfilename)
    imgfilename = 'images/papatyagray.jpeg'
    img.save(imgfilename, 'JPEG')
    arr = PIL2np(img)

    # 3x3 averaging filter
    im_out = convolve(arr,HW2a())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW2a.jpeg'
    new_img.save(imgfilename, 'JPEG')
    # 5x5 averaging filter
    im_out = convolve(arr,HW2b())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW2b.jpeg'
    new_img.save(imgfilename, 'JPEG')
    # 3x3 approximation of Gaussian filter
    im_out = convolve(arr,HW2c())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW2c.jpeg'
    new_img.save(imgfilename, 'JPEG')
    # 5x5 approximation of Gaussian filter
    im_out = convolve(arr,HW2d())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW2d.jpeg'
    new_img.save(imgfilename, 'JPEG')
    # Compute the derivative of an image in x-direction using the Sobel Operator
    im_out = convolve(arr,HW3a())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW3a.jpeg'
    new_img.save(imgfilename, 'JPEG')
    # Compute the derivative of an image in y-direction using the Sobel operatör
    im_out = convolve(arr,HW3b())
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW3b.jpeg'
    new_img.save(imgfilename, 'JPEG')

    # Compute the derivative of an image in y-direction using the Sobel operatör
    im_out1 = convolve(arr,HW3a())
    # Compute the derivative of an image in x-direction using the Sobel Operator
    im_out2 = convolve(arr,HW3b())
    #Compute the magnitude of the gradient of an image using the sobel operatör
    im_outx = np.sqrt((im_out1 * im_out1) + (im_out2 * im_out2))
    new_img = np2PIL(im_outx)
    imgfilename = 'images/HW3c.jpeg'
    new_img.save(imgfilename, 'JPEG')

    #Threshold the gradient magnitude in c.by using the following threholding function:
    #F(i, j) is 200,
    #if Grad(i, j) > = T, F(i, j) is 0, if Grad(i, j) < T
    im_out = threshold(im_outx, 45, 0,200)
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW3d1.jpeg'
    new_img.save(imgfilename, 'JPEG')
    im_out = threshold(im_outx, 60, 0,200)
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW3d2.jpeg'
    new_img.save(imgfilename, 'JPEG')
    im_out = threshold(im_outx, 90, 0,200)
    new_img = np2PIL(im_out)
    imgfilename = 'images/HW3d3.jpeg'
    new_img.save(imgfilename, 'JPEG')


#2
#By using the sample Python program in BB directory programs, read an image and compute the image convolved with

#3x3 averaging filter
def HW2a():
    return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * 1/9

#5x5 averaging filter
def HW2b():
    return np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]) * 1/25

#3x3 approximation of Gaussian filter
def HW2c():
    return np.array([[1, 2, 1 ], [2, 4, 2], [1, 2, 1]]) * 1/16

#5x5 approximation of Gaussian filter
def HW2d():
    return np.array([[1, 4, 7, 4, 1 ], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1 ]]) * 1/273

#3
#By using the sample program calculate the following gradients,

#Compute the derivative of an image in x-direction using the Sobel Operator
def HW3a():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


#Compute the derivative of an image in y-direction using the Sobel operatör
def HW3b():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def readPILimg(imgfilename):
    img = Image.open(imgfilename)
    img.show()
    img_gray = color2gray(img)
    return img_gray

def color2gray(img):
    img_gray = img.convert('L')
    return img_gray

def PIL2np(img):
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows,ncols)
    imgarray = np.array(img.convert("L"))
    return imgarray

def np2PIL(im):
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img

def convolve(im, filter):
    (nrows, ncols) = im.shape
    (k1,k2) = filter.shape
    k1h = int((k1 -1) / 2)
    k2h = int((k2 -1) / 2)
    im_out = np.zeros(shape = im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1 ):
                    if ncols > j - m and nrows > i - l:
                        sum += im[i - l][j - m] * filter[l][m]
            im_out[i][j] = sum
    return im_out

def threshold(im,T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape = im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) <  T :
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out

if __name__=='__main__':
    main()

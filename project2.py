########################################################################
#
# File:   project2.py
# Author: Ivan Lomeli
# Date:   March 2019
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This program demonstrates how to perform thresholding and morphological
# operators to video frames in order to perform blob tracking.
#
# Usage: the program can be run with a filename as a command line argument
# a command line argument.  If no argument is given, tries to capture from the
# default input 'vid1.mov'
"""
Created on Fri March 1 12:09:53 2019


"""

import numpy as np
import cv2

######################################################################
# Generates a laplacian pyramid from given image.
# Takes an 8-bit per channel RGB or grayscale image as input
# Outputs a list lp of Laplacian images (stored in numpy.float32 format)
#
# Returns a list lp
def pyr_build(img, num_layers):
    G = img.copy().astype(np.float32)
    gausPyr = [G]
    for i in range(0,num_layers):
        G = cv2.pyrDown(G).astype(np.float32)
        gausPyr.append(G)

    lp = []
    for i in range(0,num_layers):
        size = (gausPyr[i].shape[1], gausPyr[i].shape[0])
        GE = cv2.pyrUp(gausPyr[i+1], dstsize = size)
        gaus = gausPyr[i].astype(np.float32)
        enlarged = GE.astype(np.float32)
        L = cv2.subtract(gaus,enlarged)
        lp.append(L)
    lp.append(gausPyr[num_layers])
    return(lp)

######################################################################
# Reconstructs an image from its laplacian pyramid.
#
# Returns Rzero, which should be virtually indistinguishable from the original
# image
def pyr_reconstruct(lp, r0filename):
    Rlst = [lp[len(lp)-1]]
    counterNo = 0
    for i in range(len(lp)-1,0,-1):
        size = (lp[i-1].shape[1], lp[i-1].shape[0])
        enlarged = cv2.pyrUp(Rlst[counterNo], dstsize = size).astype(np.float32)
        current = enlarged + lp[i-1]
        Rlst.append(current)
        counterNo+=1
    Rlst.reverse()
    base = np.clip(Rlst[0], 0.0, 255.0)
    assert (np.amax(base) <= 255.0),"Values not restricted to the [0, 255] range"
    assert (np.amin(base) >= 0.0),"Values not restricted to the [0, 255] range"
    rzero = base.astype(np.uint8)
    cv2.imwrite(r0filename, rzero)
    return rzero

######################################################################
# This function helps combine two images with an alpha mask
#
# Returns combined images result
def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = np.expand_dims(alpha, 2)
    return A + alpha*(B-A)
######################################################################
# Labels an image by adding the specified text
#
# Returns image with added label
def label(image, text):

    # Get the image height - the first element of its shape tuple.
    h = image.shape[0]
    display = image

    # Note that even though shapes are represented as (h, w), pixel
    # coordinates below are represented as (x, y). Confusing!
    # See note below by call to cv2.ellipse

    text_pos = (16, h-16)                # (x, y) location of text
    font_face = cv2.FONT_HERSHEY_SIMPLEX # identifier of font to use
    font_size = 1.0                      # scale factor for text

    bg_color = (0, 0, 0)       # RGB color for black
    bg_size = 3                # background is bigger than foreground

    fg_color = (255, 255, 255) # RGB color for white
    fg_size = 1                # foreground is smaller than background

    line_style = cv2.LINE_AA   # make nice anti-aliased text

    # Draw background text (outline)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                bg_color, bg_size, line_style)

    # Draw foreground text (middle)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                fg_color, fg_size, line_style)
    return display
######################################################################
# This function combines each level of two image's laplacian pyramids
#
# Returns the combined laplacian pyramid
def imageCombine(lapA,lapB,mask):
    build = []
    for la,lb in zip(lapA,lapB):
        h,w,d = la.shape
        dim = (w, h)
        alphaRe= cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        current = alpha_blend(la, lb, alphaRe)
        build.append(current)
    return build

############# Combining 2 laplacian pyramids #############
#dim = (width, height)
#read in images A,B
imgA = cv2.imread('sizedA.jpeg')
height1, width1, channels = imgA.shape
dim = (width1, height1)
imgB = cv2.imread('sizedB.jpeg')
# Build Laplacian Pyramids for both
lpA = pyr_build(imgA, 6)
lpB = pyr_build(imgB, 6)
# Use 2D alpha mask to blend images
mask = np.zeros((height1,width1))
cols = mask.shape[1]
rows = mask.shape[0]
# Draw filled in mask ellipse
mask=cv2.ellipse(mask, center=(rows/2-40, cols/2+40), axes=(80,118), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
#mask=cv2.ellipse(mask, center=(cols/2, rows/2), axes=(rows/5, cols/4), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
# Save mask
cv2.imwrite('ABmask.jpeg', mask)
# alpha is a 2D floating-point array with pixel intensities in the range 0.0, 1.0
retval, dst = cv2.threshold(mask, 1.0, 1.0, cv2.THRESH_BINARY)
alpha = dst.astype(np.float32)
# Blend laplacian pyramids at each level, once using laplacian pyramids and once
# without
buildAB = imageCombine(lpA,lpB, alpha)
buildBA = imageCombine(lpB,lpA, alpha)
noPyrAB = alpha_blend(imgA, imgB, alpha)
noPyrBA = alpha_blend(imgB, imgA, alpha)
cv2.imwrite('traditionalAB.jpeg',noPyrAB)
cv2.imwrite('traditionalBA.jpeg',noPyrBA)
# Reconstruct blended laplacian pyramids and save result
ABblend = pyr_reconstruct(buildAB, "reconstructAB.jpeg")
BAblend = pyr_reconstruct(buildBA, "reconstructBA.jpeg")

################## Hybrid image #####################
# Read in image 1 and 2 (first and second)
ok1 = cv2.imread('A.jpeg')
h,w,d = ok1.shape
#dim = (width, height)
dim1 = (w, h)
first = cv2.cvtColor(ok1,cv2.COLOR_RGB2GRAY)
image1 = first.astype(np.float32)
ok2 = cv2.imread('resizedrgb2.jpeg')
# Resize if needed, make sure dim1 is smallest picture's dimensions
#second = cv2.resize(ok2, dim1, interpolation = cv2.INTER_AREA)
#cv2.imwrite('resizedrgb2.jpeg',second)
second = cv2.cvtColor(ok2,cv2.COLOR_RGB2GRAY)
image2 = second.astype(np.float32)
# Generate lopass A,B with sigmaA,sigmaB
lopassA = cv2.GaussianBlur(image1,(31,31),100).astype(np.float32)
lopassB = cv2.GaussianBlur(image2,(7,7),20).astype(np.float32)
# Generate hipass B, which is B-lopassB
hipassB = image2 - lopassB
# Convert and save
#left = lopassA.astype(np.uint8)
#cv2.imwrite('left2rgb2.jpeg', left)
#right = hipassB.astype(np.uint8)
#cv2.imwrite('right2rgb2.jpeg', right)
# Create hybrid image by adding lopassA + hipassB
result = 1*lopassA + 2*hipassB
# Keep array in range 0,255 before converting
np.clip(result, 0.0, 255.0)
result1 = result.astype(np.uint8)
# Save
cv2.imwrite('hybridrbg.jpeg', result1)

################# Going further ###############################
# Begin video
cam = cv2.VideoCapture(0)
# Label display window
cv2.namedWindow("Smile")

# Container to hold eventual selfie
imgC = imgA.copy()
img_counter = 0
# Until user hits escape key
while True:
    # Read current frame
    ret, frame = cam.read()
    # Copy dimensions
    height2, width2, channels = frame.shape
    # Build mask with same dimensions
    mask = np.zeros((height2,width2))
    cols = mask.shape[1]
    rows = mask.shape[0]
    # Draw ellipse to prompt user face area
    cv2.ellipse(frame, center=(cols/2, rows/2+10), axes=(rows/4+10, cols/6+3), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=1)
    # Label frame with user prompt
    prompt = label(frame, 'Hold face inside oval, hit space and hold for 3 seconds. Press Esc once done')
    # If user hits spacebar and we fail to record frame
    cv2.imshow("smile", prompt)
    if not ret:
        break
    k = cv2.waitKey(1)
    # User hits Esc key
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    #user presses SPACE key
    elif k%256 == 32:
        # SPACE pressed
        # Save image
        img_name = "selfie{}.jpeg".format(img_counter)
        # Set container value to this frame
        imgC = frame
        # Output saved filename
        print("{} written!".format(img_name))
        img_counter += 1
#Exit
cam.release()
cv2.destroyAllWindows()

#cv2.imwrite('selfie2.png', imgC)

heightC, widthC, channelsC = imgA.shape
# Resize to fit image A from laplacian image blend above
dim = (width1, height1)
imgC = cv2.resize(imgC, dim, interpolation = cv2.INTER_AREA)
# Save resized image
cv2.imwrite('resizedCpic2.jpeg',imgC)
# Build laplacian pyramid for selfie image
lpC = pyr_build(imgC, 6)
# Build mask for selfie image
mask = np.zeros((height1,width1))
cols = mask.shape[1]
rows = mask.shape[0]
# Draw ellipse in mask
mask=cv2.ellipse(mask, center=(cols/2, rows/2+5), axes=(rows/8-3, cols/3-10), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
# Save mask
cv2.imwrite('mask2.jpeg', mask)
retval, dst = cv2.threshold(mask, 1.0, 1.0, cv2.THRESH_BINARY)
alpha = dst.astype(np.float32)
# Combine laplacian pyramids of image A and C
buildAC = imageCombine(lpA,lpC, alpha)
buildBC = imageCombine(lpB,lpC, alpha)
# Blend two imags without laplacian pyramids
noPyrAC = alpha_blend(imgA, imgC, alpha)
noPyrBC = alpha_blend(imgB, imgC, alpha)
# Save results
cv2.imwrite('traditionalBlendSelfieAC.jpeg',noPyrAC)
cv2.imwrite('traditionalBlendSelfieBC.jpeg',noPyrBC)
blendAC = pyr_reconstruct(buildAC, "reconstructAC.jpeg")
blendBC = pyr_reconstruct(buildBC, "reconstructBC.jpeg")

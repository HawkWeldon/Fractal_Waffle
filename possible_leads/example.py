import cv2 as cv
from scipy.fftpack import dct, idct
import numpy as np
from PIL import Image

image = cv.imread("test.png")
# image_dct = cv.imread("test.png")
# image_quant = cv.imread("test.png")

x = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
arr = np.asarray(x, float)
image_quant = np.asarray(x, float)

#Quantization matrix for jpeg standard
qm = [[16,    11,    10,    16,    24,    40,    51,    61],
    [12,    12,    14,    19,    26,    58,    60,    55],
    [14,    13,    16,    24,    40,    57,    69,    56],
    [14,    17,    22,    29,    51,    87,    80,    62],
    [18,    22,    37,    56,    68,   109,   103,    77],
    [24,    35,    55,    64,    81,   104,   113,    92],
    [49,    64,    78,    87,   103,   121,   120,   101],
    [72,    92,    95,    98,   112,   100,   103,    99]]

# 8x8 jpeg window size
ws_r = 8
ws_c = 8

# Method/function to quantize each element of the DCTed matrix 8x8 by the 
# standard 50% qm matrix
def quantize(inMatrix, qm):
    outMatrix = np.empty((8, 8))

    for i in range(0, 8):
        for j in range(0, 8):
            # outMatrix[i][j] = np.linalg.norm(inMatrix[i][j]/qm[i][j])
            # outMatrix[i][j] = np.round(inMatrix[i][j]/qm[i][j])
            outMatrix[i][j] = inMatrix[i][j]/qm[i][j]
    # print(outMatrix)
    return outMatrix

# Devide the image into 8x8 blocks and apply to each block DCT
for r in range(0, np.size(arr, 0)-ws_r, ws_r):
    for c in range(0, np.size(arr, 1)-ws_c, ws_c):
        window = arr[r:r+ws_r, c:c+ws_c]
        # print(dct(window))
        image_quant[r:r+ws_r, c:c+ws_c] = quantize(dct(window),qm)

image_quant = Image.fromarray(image_quant)
image_quant = image_quant.convert('RGB')
image_quant.save('test_quant_dct.png')
image_quant.show()

#Now invert the above process by dequantizing and applying inverce DCT
image = cv.imread("test_quant_dct.png")

x = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
arr = np.asarray(x, float)
image_dequant = np.asarray(x, float)

def dequantize(inMatrix, qm):
    outMatrix = np.empty((8, 8))

    for i in range(0, 8):
        for j in range(0, 8):
            # outMatrix[i][j] = np.linalg.norm(inMatrix[i][j]/qm[i][j])
            # outMatrix[i][j] = np.round(inMatrix[i][j]*qm[i][j])
            outMatrix[i][j] = inMatrix[i][j]*qm[i][j]
    # print(outMatrix)
    return outMatrix

for r in range(0, np.size(arr, 0)-ws_r, ws_r):
    for c in range(0, np.size(arr, 1)-ws_c, ws_c):
        window = arr[r:r+ws_r, c:c+ws_c]
        # image_dequant[r:r+ws_r, c:c+ws_c] = dequantize(idct(window),qm)
        image_dequant[r:r+ws_r, c:c+ws_c] = idct(dequantize(window,qm))

image_dequant = Image.fromarray(image_dequant)
image_dequant = image_dequant.convert('RGB')
image_dequant.save('test_dequant_dct.png')
image_dequant.show()
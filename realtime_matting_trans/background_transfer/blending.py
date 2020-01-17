import cv2
import numpy as np
from matplotlib import  pyplot as plt

# Read images : obj image will be cloned into im
obj = cv2.imread("inputs/contents/face.jpg")
im= cv2.imread("inputs/styles/99.jpg")
H,W = obj.shape[:2]
obj = cv2.resize(obj, (100, 100), cv2.INTER_CUBIC)
# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the obj in the im
width, height, channels = im.shape

print(obj.shape, im.shape)

center = (int(height/2), int(width/2))

# Seamlessly clone obj into im and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
normal_clone=normal_clone[:, :, (2, 1, 0)]
plt.imshow(normal_clone)
plt.show()

mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)
plt.imshow(mixed_clone)
plt.show()

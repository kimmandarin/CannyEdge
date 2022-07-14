import cv2
import numpy as np
import matplotlib.pyplot as plt

img3 = cv2.imread('./img/002.jpg', cv2.IMREAD_GRAYSCALE)
canny_50 = cv2.Canny(img3, 50, 200)
canny_100 = cv2.Canny(img3, 100, 200)
canny_170 = cv2.Canny(img3, 170, 200)

titles = ['original', 'canny_50', 'canny_100', 'canny_170']
images = [img3, canny_50, canny_100, canny_170]
cv2.imshow('original', img3)
cv2.imshow('canny_50', canny_50)
cv2.imshow('canny_100', canny_100)
cv2.imshow('canny_170', canny_170)
#cv2.waitkey(0)
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
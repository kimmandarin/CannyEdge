import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./img/006.jpg', cv2.IMREAD_GRAYSCALE)
destroyNoise = cv2.GaussianBlur(image, (5, 5), 0.3)

Gx = cv2.Sobel(np.float32(destroyNoise), cv2.CV_32F, 1, 0, 3)
Gy = cv2.Sobel(np.float32(destroyNoise), cv2.CV_32F, 0, 1, 3)

sobel = cv2.magnitude(Gx, Gy)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

titles = ['destroyNoise', 'sobel']
images = [destroyNoise, sobel]
cv2.imshow('destroyNoise', destroyNoise)
cv2.imshow('sobel', sobel)
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
for i in range(2):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
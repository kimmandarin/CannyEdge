import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./img/002.jpg', cv2.IMREAD_GRAYSCALE)
destroyNoise = cv2.GaussianBlur(image, (5, 5), 0.3)

titles = ['original', 'destroyNoise']
images = [image, destroyNoise]
cv2.imshow('original', image)
cv2.imshow('destroyNoise', destroyNoise)
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
for i in range(2):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
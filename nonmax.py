import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./img/002.jpg', cv2.IMREAD_GRAYSCALE)
destroyNoise = cv2.GaussianBlur(image, (5, 5), 0.3)

Gx = cv2.Sobel(np.float32(destroyNoise), cv2.CV_32F, 1, 0, 3)
Gy = cv2.Sobel(np.float32(destroyNoise), cv2.CV_32F, 0, 1, 3)

sobel = cv2.magnitude(Gx, Gy)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)


def nonmax_suppression(sobel, direct):
    rows, cols = sobel.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            values = sobel[i - 1:i + 2, j - 1:j + 2].flatten()
            first = [3, 0, 1, 2]
            id = first[direct[i, j]]
            v1, v2 = values[id], values[8 - id]
            dst[i, j] = sobel[i, j] if (v1 < sobel[i, j] > v2) else 0
    return dst

directs = cv2.phase(Gx, Gy) / (np.pi / 4)
directs = directs.astype(int) % 4
max_sobel = nonmax_suppression(sobel, directs)

titles = ['sobel', 'max_sobel']
images = [sobel, max_sobel]
cv2.imshow('sobel', max_sobel)
cv2.imshow('sobel', max_sobel)
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
for i in range(2):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
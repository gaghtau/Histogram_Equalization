import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "image13_1736425790.jpg"  
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Не удалось открыть {image_path}")

# --- Цветное изображение и гистограммы BGR ---
b_channel, g_channel, r_channel = cv2.split(image)
channels = [b_channel, g_channel, r_channel]
colors = ["b", "g", "r"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for chan, color in zip(channels, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title("Гистограммы BGR каналов")
plt.xlabel("Значение пикселя")
plt.ylabel("Количество пикселей")

# --- Серое изображение и выравнивание гистограммы ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)

plt.subplot(1, 2, 2)
plt.hist(gray.ravel(), bins=256, color='gray', alpha=0.5, label='Оригинал')
plt.hist(equalized.ravel(), bins=256, color='black', alpha=0.5, label='Эквализация')
plt.title("Гистограмма серого изображения")
plt.xlabel("Значение пикселя")
plt.ylabel("Количество пикселей")
plt.legend()

plt.tight_layout()
plt.show()

# --- Показываем изображения ---
cv2.imshow("Оригинал", image)
cv2.imshow("Серое с выравниванием", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Завантаження вхідного зображення
img_path = "Image_people.jpg"  # Вкажіть правильний шлях до файлу
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Помилка: зображення не знайдено!")

# Перетворення формату кольорів на RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Виконання корекції яскравості за допомогою LAB-простору
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(img_lab)
l_adjusted = cv2.equalizeHist(l_channel)
adjusted_img = cv2.merge((l_adjusted, a_channel, b_channel))
adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_LAB2BGR)

# Підготовка даних для кластеризації
pixels = adjusted_img.reshape((-1, 3)).astype(np.float32)

# Виконання кластеризації методом K-Means
clusters = 5  # Визначена кількість кластерів
kmeans = KMeans(n_clusters=clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(pixels)
kmeans_centroids = np.uint8(kmeans.cluster_centers_)
kmeans_result = kmeans_centroids[kmeans_labels.flatten()]
kmeans_result = kmeans_result.reshape(img.shape)

# Перетворення кластеризованого зображення у відтінки сірого
gray_image = cv2.cvtColor(kmeans_result, cv2.COLOR_BGR2GRAY)

# Виділення контурів на сегментованому зображенні
_, binary_thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Накладання контурів на вихідне зображення
final_output = img_rgb.copy()
cv2.drawContours(final_output, contours, -1, (255, 0, 0), 2)

# Візуалізація отриманих результатів
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Початкове зображення")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Сегментоване зображення")
plt.imshow(kmeans_result)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Контури об'єкта")
plt.imshow(final_output)
plt.axis("off")

plt.show()

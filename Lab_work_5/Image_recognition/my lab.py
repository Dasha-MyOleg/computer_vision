import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Завантаження зображення
image_path = "Image_people.jpg"  # Замініть на шлях до вашого зображення
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом")

# Зміна кольорового простору на RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Покращення якості зображення (корекція гістограми)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(image_lab)
l_eq = cv2.equalizeHist(l)
image_eq = cv2.merge((l_eq, a, b))
image_eq = cv2.cvtColor(image_eq, cv2.COLOR_LAB2BGR)

# Перетворення зображення у 2D матрицю для кластеризації
pixel_values = image_eq.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Кластеризація методом k-means
k = 5  # Кількість кластерів
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Перетворення результату кластеризації в градації сірого
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Виділення контурів
_, thresh = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Відображення контурів на оригінальному зображенні
output_image = image_rgb.copy()
cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

# Візуалізація результатів
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Оригінальне зображення")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Сегментоване зображення")
plt.imshow(segmented_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Контури об'єкта")
plt.imshow(output_image)
plt.axis("off")

plt.show()

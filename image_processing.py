import cv2
import numpy as np
import matplotlib.pyplot as plt

# Зчитування зображення
image = cv2.imread('image/I22.BMP')  # Зчитуємо картинку (Task 1)

def bgr_to_rgb(image_bgr):  # BGR -> RGB
    image_rgb = image_bgr.copy()
    image_rgb[..., 0] = image_bgr[..., 2]  # R
    image_rgb[..., 2] = image_bgr[..., 0]  # B
    return image_rgb

def calculate_shannon_entropy(image):
    # Перетворення зображення в одномірний масив
    pixels = image.flatten()
    
    # Обчислення гістограми значень пікселів
    histogram, _ = np.histogram(pixels, bins=256, range=(0, 256))
    
    # Обчислення ймовірностей
    probabilities = histogram / histogram.sum()
    
    # Вилучення нульових ймовірностей
    probabilities = probabilities[probabilities > 0]
    
    # Обчислення ентропії Шенона
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_hartley_measure(image):
    # Кількість можливих значень пікселів (для 8-бітних зображень)
    num_possible_values = 256
    
    # Міра Хартлі
    hartley_measure = np.log2(num_possible_values)
    
    return hartley_measure

def calculate_markov_process(image):
    # Перетворення зображення в одномірний масив
    pixels = image.flatten()
    
    # Створення матриці переходів
    transition_matrix = np.zeros((256, 256))
    
    # Розрахунок ймовірностей переходів між сусідніми пікселями
    rows, cols = image.shape[:2]
    
    for i in range(rows):
        for j in range(cols):
            current_pixel = pixels[i * cols + j]
            
            # Сусіди: праворуч, внизу
            if j < cols - 1:  # праворуч
                right_pixel = pixels[i * cols + (j + 1)]
                transition_matrix[current_pixel, right_pixel] += 1
            
            if i < rows - 1:  # внизу
                below_pixel = pixels[(i + 1) * cols + j]
                transition_matrix[current_pixel, below_pixel] += 1

    # Нормалізація матриці переходів
    transition_matrix += 1e-10  # Додаємо мале значення для уникнення ділення на нуль
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

# Вивід оригінального зображення
image_task1 = bgr_to_rgb(image)
plt.imshow(image_task1)
plt.title('Selected image (Task 1)')
plt.axis('off')  # Вимкнення осей
plt.show()

# Обчислення ентропії, міри Хартлі та матриці переходів
entropy = calculate_shannon_entropy(image)
hartley_measure = calculate_hartley_measure(image)
transition_matrix = calculate_markov_process(image)

print(f"Ентропія Шенона: {entropy}")
print(f"Міра Хартлі: {hartley_measure}")
print("Матриця переходів (перших 5 значень):")
print(transition_matrix[:5, :5])  # Вивід лише частини матриці для зручності

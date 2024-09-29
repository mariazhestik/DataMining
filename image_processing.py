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
    pixels = image.flatten()
    histogram, _ = np.histogram(pixels, bins=256, range=(0, 256))
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_hartley_measure(image):
    num_possible_values = 256
    hartley_measure = np.log2(num_possible_values)
    return hartley_measure

def calculate_markov_process(image):
    pixels = image.flatten()
    transition_matrix = np.zeros((256, 256))
    rows, cols = image.shape[:2]
    
    for i in range(rows):
        for j in range(cols):
            current_pixel = pixels[i * cols + j]
            
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

# Додатковий новий функціонал
def segment_image(image, n_segments=4):
    height = image.shape[0]
    segments = np.array_split(image, n_segments, axis=0)
    return segments

def process_segments(segments):
    entropy_values = []
    hartley_values = []
    markov_matrices = []
    
    for segment in segments:
        entropy = calculate_shannon_entropy(segment)
        hartley = calculate_hartley_measure(segment)
        markov_matrix = calculate_markov_process(segment)
        
        entropy_values.append(entropy)
        hartley_values.append(hartley)
        markov_matrices.append(markov_matrix)
    
    return entropy_values, hartley_values, markov_matrices

def compare_results(entropy_values, hartley_values):
    avg_entropy = np.mean(entropy_values)
    avg_hartley = np.mean(hartley_values)
    
    print(f"Середнє значення ентропії для всіх сегментів: {avg_entropy}")
    print(f"Середнє значення міри Хартлі для всіх сегментів: {avg_hartley}")
    
    return avg_entropy, avg_hartley

def visualize_results(image, entropy_values, hartley_values, markov_matrices):
    n_segments = len(entropy_values)
    fig, ax = plt.subplots(1, n_segments, figsize=(15, 5))
    
    segments = np.array_split(image, n_segments, axis=0)
    
    for i, segment in enumerate(segments):
        ax[i].imshow(bgr_to_rgb(segment))
        ax[i].set_title(f"Сегмент {i+1}\nЕнтропія: {entropy_values[i]:.2f}\nХартлі: {hartley_values[i]:.2f}")
        ax[i].axis('off')

    plt.show()
    
    # Виведення Марковських матриць для кожного сегмента
    for i, matrix in enumerate(markov_matrices):
        print(f"Марковська матриця для сегмента {i + 1}:")
        print(matrix[:5, :5])  # Виводимо тільки перші 5x5 елементів для зручності
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.title(f'Markov Matrix for Segment {i + 1}')
        plt.colorbar()
        plt.show()

# Вивід оригінального зображення
image_task1 = bgr_to_rgb(image)
plt.imshow(image_task1)
plt.title('Selected image (Task 1)')
plt.axis('off')
plt.show()

# Сегментація зображення
segments = segment_image(image_task1, n_segments=4)

# Обчислення значень для кожного сегмента
entropy_values, hartley_values, markov_matrices = process_segments(segments)

# Порівняння середніх значень
avg_entropy, avg_hartley = compare_results(entropy_values, hartley_values)

# Візуалізація результатів
visualize_results(image_task1, entropy_values, hartley_values, markov_matrices)

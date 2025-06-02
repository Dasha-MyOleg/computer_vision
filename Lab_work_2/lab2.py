import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import argparse
import matplotlib.pyplot as plt
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN)


#input_path = r"C:\Users\Даша\OneDrive\Документы\GitHub\computer_vision_1\Lab_work_2\image.png"
#output_path = r"C:\Users\Даша\OneDrive\Документы\GitHub\computer_vision_1\Lab_work_2\result.png"

def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"Помилка: Файл {image_path} не знайдено!")
        exit()
    return np.array(Image.open(image_path))



def save_image(image_array, output_path, format="PNG"):
    output_image = Image.fromarray(image_array.clip(0, 255).astype(np.uint8))
    output_image.save(output_path, format=format)


def get_gradient_factor(x, y, width, height, gradient_type):
    factor = 1.0
    if gradient_type == 'diagonal_lr':
        factor = (x + y) / (width + height)
    elif gradient_type == 'diagonal_rl':
        factor = (width - x + y) / (width + height)
    elif gradient_type == 'diagonal_tb':
        factor = y / height
    elif gradient_type == 'diagonal_bt':
        factor = (height - y) / height
    elif gradient_type == 'from_center':
        factor = ((x - width // 2) ** 2 + (y - height // 2) ** 2) ** 0.5 / ((width ** 2 + height ** 2) ** 0.5)
    elif gradient_type == 'to_center':
        factor = 1 - ((x - width // 2) ** 2 + (y - height // 2) ** 2) ** 0.5 / ((width ** 2 + height ** 2) ** 0.5)
    return max(0.1, min(1.0, factor))



def select_option(options, prompt):
    print(prompt)
    for key, value in options.items():
        print(f"{key} - {value}")
    choice = input("Введіть номер: ").strip()
    while choice not in options:
        print("Невірний вибір, спробуйте ще раз.")
        choice = input("Введіть номер: ").strip()
    return options[choice]


def apply_grayscale(image, gradient_type):
    image_pil = Image.fromarray(image).convert('RGB')  # Конвертація для редагування
    pix = image_pil.load()
    width, height = image_pil.size
    gray_image = Image.new("RGB", (width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j]
            gray = (r + g + b) // 3
            factor = get_gradient_factor(i, j, width, height, gradient_type)
            new_gray = int(gray * factor)
            gray_image.putpixel((i, j), (new_gray, new_gray, new_gray))

    return np.array(gray_image)



def apply_negative(image, gradient_type):
    image_pil = Image.fromarray(image).convert('RGB')  # Конвертація для можливості редагування
    pix = image_pil.load()
    width, height = image_pil.size
    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j]
            factor = get_gradient_factor(i, j, width, height, gradient_type)
            pix[i, j] = (
                int((255 - r) * factor),
                int((255 - g) * factor),
                int((255 - b) * factor)
            )
    return np.array(image_pil)


def apply_monochrome(image, gradient_type):
    image_pil = Image.fromarray(image).convert('RGB')
    pix = image_pil.load()
    width, height = image_pil.size
    monochrome_image = Image.new("RGB", (width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j]
            gray = (r + g + b) // 3
            threshold = 128  # Фіксований поріг для чорно-білого зображення
            factor = get_gradient_factor(i, j, width, height, gradient_type)
            if gray * factor > threshold:
                monochrome_image.putpixel((i, j), (255, 255, 255))
            else:
                monochrome_image.putpixel((i, j), (0, 0, 0))

    return np.array(monochrome_image)


def apply_noise(image):
    noise_level = 300  # Регульований рівень шуму
    noisy_image = image.copy()
    height, width, channels = image.shape
    noise = np.random.randint(-noise_level, noise_level, (height, width, channels))
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_sepia(image, gradient_type):
    image_pil = Image.fromarray(image).convert('RGB')
    pix = image_pil.load()
    width, height = image_pil.size
    sepia_image = Image.new("RGB", (width, height))

    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j]
            factor = get_gradient_factor(i, j, width, height, gradient_type)
            tr = int((0.393 * r + 0.769 * g + 0.189 * b) * factor)
            tg = int((0.349 * r + 0.686 * g + 0.168 * b) * factor)
            tb = int((0.272 * r + 0.534 * g + 0.131 * b) * factor)
            sepia_image.putpixel((i, j), (min(255, tr), min(255, tg), min(255, tb)))

    return np.array(sepia_image)



def apply_brightness(image, gradient_type, factor):
    return np.clip(image + factor, 0, 255).astype(np.uint8)


def apply_filter(image, filter_type):
    image_pil = Image.fromarray(image)
    if filter_type == "contour":
        image_pil = image_pil.filter(ImageFilter.CONTOUR)
    elif filter_type == "sharpen":
        image_pil = image_pil.filter(ImageFilter.SHARPEN)
    return np.array(image_pil)

def main():
    parser = argparse.ArgumentParser(description='Image color correction script.')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--output', required=True, help='Path to save processed image')
    args = parser.parse_args()

    image = load_image(args.input)

    effects = select_option({
        "0": "grayscale",
        "1": "sepia",
        "2": "negative",
        "3": "noise",
        "4": "brightness",
        "5": "monochrome",
        "6": "contour",
        "7": "gradient"
    }, "Оберіть тип перетворення!")

    brightness_factor = 0
    gradient_type = ""

    if effects == "brightness":
        brightness_factor = int(input("Введіть фактор яскравості (-100 до 100): "))

    if effects == "gradient":
        gradient_type = select_option({
            "0": "diagonal_lr",
            "1": "diagonal_rl",
            "2": "diagonal_tb",
            "3": "diagonal_bt",
            "4": "from_center",
            "5": "to_center"
        }, "Оберіть тип градієнта!")
        effects = select_option({
            "0": "grayscale",
            "1": "sepia",
            "2": "negative"
        }, "Оберіть ефект для застосування з градієнтом:")

    if effects == "grayscale":
        result = apply_grayscale(image, gradient_type)
    elif effects == "negative":
        result = apply_negative(image, gradient_type)
    elif effects == "sepia":
        result = apply_sepia(image, gradient_type)
    elif effects == "brightness":
        result = apply_brightness(image, gradient_type, brightness_factor)
    elif effects == "contour":
        result = apply_filter(image, "contour")
    elif effects == "monochrome":
        result = apply_monochrome(image, gradient_type)
    elif effects == "noise":
        result = apply_noise(image)
    else:
        print("Невідомий ефект!")
        return

    save_image(result, args.output)
    print(f"Processed image saved at {args.output}")

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(Image.open(args.input))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed')
    plt.imshow(result)
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()



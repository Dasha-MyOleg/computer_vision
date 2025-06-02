import os

cascade_path = r"C:\Users\Даша\OneDrive\Документы\GitHub\computer_vision_1\Lab_work_5\Image_recognition\haarcascade_russian_plate_number.xml"

if os.path.exists(cascade_path):
    print("Файл знайдено.")
else:
    print("Файл не знайдено.")

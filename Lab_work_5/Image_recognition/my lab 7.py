import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model


cascade_path = "C:\\Users\\Даша\\OneDrive\\Документы\\GitHub\\computer_vision_1\\Lab_work_5\\Image_recognition\\haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)


# Step 1: Load Pre-trained Model (if any, or placeholder for future use)
def load_license_plate_model():
    """Load a pre-trained model for license plate recognition."""
    try:
        model = load_model("license_plate_model.h5")
        return model
    except Exception as e:
        print("Error loading model: ", e)
        return None


# Step 2: Pre-process Image
def preprocess_image(image):
    """Convert image to grayscale and apply thresholding for better OCR performance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# Step 3: Detect License Plate (Haar Cascade as example)
def detect_license_plate(frame, cascade_path="haarcascade_russian_plate_number.xml"):
    """Detect license plate in the image using Haar Cascade."""
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return plates


# Step 4: Extract and Recognize Text
def recognize_license_plate_text(image):
    """Recognize text from the license plate area using Tesseract."""
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Adjust for your Tesseract path
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()


# Step 5: Process Video Stream
def process_video(video_path):
    """Process the video frame by frame to detect and recognize license plates."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        plates = detect_license_plate(frame)
        for (x, y, w, h) in plates:
            plate_area = frame[y:y + h, x:x + w]
            processed_plate = preprocess_image(plate_area)
            text = recognize_license_plate_text(processed_plate)

            # Draw the rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('License Plate Recognition', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Function (Example Usage)
if __name__ == "__main__":
    video_path = "C:/MyProjects/lab7video.mp4"  # Path to the video file
    process_video(video_path)


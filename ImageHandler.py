import cv2
import easyocr
import os
from tqdm import tqdm

def read_images():
    """
    Read images from the input directory, extract text using EasyOCR, and save the extracted text
    into text files in the output directory.

    This function leverages EasyOCR for text extraction and supports common image formats.
    """
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=True)  # Set `gpu=True` if a GPU is available for faster processing

    # Input directory containing image files
    input_directory = 'DataImages'

    # Output folder for text files
    output_folder = 'readTextFiles'

    # Supported image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all image files in the input directory
    image_files = [
        file_name for file_name in os.listdir(input_directory)
        if file_name.lower().endswith(image_extensions)
    ]

    # Loop through all image files with a progress bar
    for file_name in tqdm(image_files, desc="Processing images", unit="image"):
        # Construct the full path to the image
        image_path = os.path.join(input_directory, file_name)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale (optional, improves OCR accuracy)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the image
        result = reader.readtext(gray, detail=0)  # `detail=0` returns only the recognized text

        # Combine extracted text into a single string
        text = '\n'.join(result)

        # Generate the output file name
        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")

        # Save the extracted text to the file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)

    print("All images processed. Extracted text saved to the 'readTextFiles' folder.")

# Example usage
if __name__ == "__main__":
    read_images()

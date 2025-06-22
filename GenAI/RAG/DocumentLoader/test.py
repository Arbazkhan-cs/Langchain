from PIL import Image
import pytesseract

text = pytesseract.image_to_string(Image.open("converted_page1.jpg"))
print(text)

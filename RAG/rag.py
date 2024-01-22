from transformers import pipeline
from PIL import Image
from pytesseract import pytesseract

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

dl_path = r'example_data\data_lakes.png'
table_path = r'example_data\table.png'
'''img = Image.open(table_path)
img.show()
text = pytesseract.image_to_string(img)
print(text)'''

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)
# Extract answer from text
output_dl = nlp(
    image=dl_path,
    question="What are Big Data Lakes?",
)
output_table = nlp(
    image=table_path,
    question="What is the industry with more numbers of employees?",
)

print(output_dl)
# [{'score': 0.762671172618866,
# 'answer': 'natural evolution of data warehousing systems',
# 'start': 9, 'end': 14}]
print(output_table)
# [{'score': 0.7530626654624939,
#  'answer': 'Sporting Goods / Retail',
#  'start': 37, 'end': 40}]
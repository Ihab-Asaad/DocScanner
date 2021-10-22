# DocScanner
Extract text from .pdf files or from images containing documents.
# Description:
Create a command-line tool to extract text from images containing documents or from .pdf files.
# Input:
Documents in images or .pdf files must contain English only(for now). Supported fomats:
-	PNG
-	JPEG
-	PDF
# Output:
Extracted text as a text file.
# Interface:
**Python main.py --input=./inputImg.jpg --output=extractedText.txt --verbose** , 
Where:
-	Input : input file
-	Output : output text file
-	Verbose: verbose mode (output detailed logs)
# INSTALLATION:
- See https://pypi.org/project/pytesseract/ to install Tesseract-OCR from Prerequisites, or directly from https://tesseract-ocr.github.io/tessdoc/Compiling.html. For example, if using Linux, run the following command : **sudo apt install tesseract-ocr** To install Tesseract 4.x
-	To install and setup the dependencies, run:  **pip install -r requirements.txt**
# Example:
- 'inputImg.jpg' is an input image example captured from phone, and 'extractedText.txt' is the output text.

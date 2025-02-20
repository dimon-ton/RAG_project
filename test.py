import fitz  # PyMuPDF


def save_pdf_text_to_file(pdf_path, txt_file_path):
    doc = fitz.open(pdf_path)
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        for page in doc:
            text = page.get_text()
            file.write(text + '\n')

# Usage:
save_pdf_text_to_file("data/pdf_files/myPDF.pdf", "output.txt")
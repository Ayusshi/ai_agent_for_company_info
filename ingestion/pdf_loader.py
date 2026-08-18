import pymupdf
import os


def load_pdf(file_path):
    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "source": os.path.basename(file_path)
            })

    document.close()

    return pages
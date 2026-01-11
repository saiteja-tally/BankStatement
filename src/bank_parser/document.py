from .utils import is_pdf_scanned, is_garbage_text, normalize_pdf_to_a4_pdfplumber
import pdfplumber
import os
from pdfminer.pdfdocument import (
        PDFPasswordIncorrect,
        PDFTextExtractionNotAllowed,
    )
from typing import Generator, Tuple, Optional

class Document:
    def __init__(self, pdf_path, password=None):
        self.is_scanned = is_pdf_scanned(pdf_path, password=password)
        self.pdf_path = pdf_path
        self.password = password  
        self.plumber_doc = None
        self.pages = []
        self.table_v_lines = []
        self.table_headers = []

    def open_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError("File path is invalid.")

        if not self.pdf_path.lower().endswith('.pdf'):
            raise ValueError("The file does not have a .pdf extension.")

        with open(self.pdf_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                raise ValueError("The file is not a valid PDF.")

        # Try opening with password (None is allowed), map common errors to clearer messages
        try:
            self.plumber_doc = pdfplumber.open(self.pdf_path, password=self.password)
            if is_garbage_text(self.plumber_doc.pages[0].extract_text()):
                normalized_pdf_path = normalize_pdf_to_a4_pdfplumber(self.pdf_path, dpi=300, password=self.password)
                self.plumber_doc = pdfplumber.open(normalized_pdf_path, password=self.password)
            else:
                self.plumber_doc = pdfplumber.open(self.pdf_path, password=self.password)
        except PDFPasswordIncorrect:
            raise ValueError("Incorrect password provided.")
        except PDFTextExtractionNotAllowed:
            # File opened but extraction not permitted by the document's settings
            raise PermissionError("Text extraction is not allowed for this PDF.")
        except Exception as e:
            # Surface other errors unchanged
            raise

    def stream_pages(self, start: int = 0, end: Optional[int] = None) -> Generator[Tuple[int, pdfplumber.page.Page], None, None]:
        """
        Yield lightweight `pdfplumber.page.Page` objects without wrapping them in `Page`.
        Consumer should rasterize on demand and free buffers as soon as possible.
        """
        if end is None:
            end = len(self.plumber_doc.pages)
        if start < 0 or end > len(self.plumber_doc.pages):
            raise IndexError("start/end out of range")
        for i in range(start, end):
            yield i, self.plumber_doc.pages[i]
    

    def following_blueprint(self, page) -> bool:
        """
        Check if the current page follows the same table blueprint as previous pages.
        """


        ## Check the vertical lines for each page that are at same X or very close as compared to page 1, if a page has very different vertical lines than page 1, the page should be discarded
        
        matched_lines = 0
        for line in page.table_v_lines:
            for ref_line in self.table_v_lines:
                if abs(((line[0] + line[2]) / 2) - ((ref_line[0] + ref_line[2]) / 2)) <= 0.014:
                    matched_lines += 1
                    break
        # If less than 50% of the lines match, discard the page
        if (matched_lines / len(self.table_v_lines)) < 0.5:
            return False

        else:
            return True

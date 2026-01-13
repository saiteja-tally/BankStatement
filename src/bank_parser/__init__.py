import os

from .table_detect import TableDetector
from .tsr import TableRecognizer
from .exceptions import ScannedPDFNotSupported
from .ocr import OCREngine
from .document import Document
from .page import Page
from .config import is_debug_mode

class BankStatementParser:
    def __init__(self):   
        self.recognizer = TableRecognizer()
        self.detector = TableDetector(self.recognizer)
        self.ocr_engine = OCREngine()

    def parse(self, pdf_path, output, password: str = None):
        
        Doc = Document(pdf_path, password)

        if Doc.is_scanned:
            raise ScannedPDFNotSupported("The provided PDF is a scanned document.")

        Doc.open_pdf()

        transactions = []

        for _, plumber_page in Doc.stream_pages():
            page = Page(plumber_page.page_number)
            page.table_xyxy, page.table_h_lines, page.table_v_lines, page.table_type = self.detector.get_table(plumber_page)
            page.table_ocr_data = self.ocr_engine.extract_table_ocr_data(plumber_page, page.table_xyxy) if page.table_xyxy else None
            page.table_headers = self.recognizer.recognize_table_headers(page) if page.table_h_lines else []

            if is_debug_mode():
                print(f"Page {plumber_page.page_number}: Detected table type: {page.table_type}")
                image = plumber_page.to_image(resolution=300).original
                width, height = image.size

                temp_output = os.path.join(output, os.path.basename(pdf_path).replace('.pdf', ''))
                os.makedirs(temp_output, exist_ok=True)
                if page.table_xyxy:
                    from PIL import Image, ImageDraw, ImageFont
                    draw = ImageDraw.Draw(image)
                    x0, y0, x1, y1 = page.table_xyxy
                    draw.text([x0 * width, y0 * height - 25], f"Table( {page.table_type})", fill="red", font=ImageFont.load_default(size=20))
                    draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline="red", width=5)

                if page.table_h_lines:
                    for line in page.table_h_lines:
                        x0, y0, x1, y1 = line
                        draw.text([x0 * width-10, y0 * height], "H", fill="blue")
                        draw.line([x0 * width, y0 * height, x1 * width, y1 * height], fill="blue", width=3)
                if page.table_v_lines:
                    for line in page.table_v_lines:
                        x0, y0, x1, y1 = line
                        draw.text([x0 * width, y0 * height - 10], "V", fill="green")
                        draw.line([x0 * width, y0 * height, x1 * width, y1 * height], fill="green", width=3)

                if page.table_ocr_data:
                    for ocr_text in page.table_ocr_data:
                        x0 = ocr_text.bbox_xyxy[0] * width
                        y0 = ocr_text.bbox_xyxy[1] * height
                        x1 = ocr_text.bbox_xyxy[2] * width
                        y1 = ocr_text.bbox_xyxy[3] * height
                        
                        draw.rectangle([x0, y0, x1, y1], outline="purple", width=1)

                image.save(f"{temp_output}/page_{plumber_page.page_number}.png")
            
            Doc.pages.append(page)

        for page in Doc.pages:
            if not page.table_v_lines:
                continue
            if not Doc.table_headers and page.table_headers and page.table_v_lines:
                Doc.table_v_lines = page.table_v_lines
                Doc.table_headers = page.table_headers
            
            if not Doc.table_headers:
                continue

            if Doc.following_blueprint(page):
                page.table_headers = Doc.table_headers
                page_transactions = self.recognizer.extract_transactions_from_page(page)
                if is_debug_mode():
                    print(f"Page {page.page_number}: Extracted {len(page_transactions)} transactions based on blueprint.")
                    with open(os.path.join(temp_output, f"page_{page.page_number}_transactions.json"), "w") as f:
                        import json
                        json.dump(page_transactions, f, indent=4)

                transactions.extend(page_transactions)
            else:
                print(f"Page {page.page_number}: Table structure differs significantly from blueprint. Skipping transaction extraction.")

        if is_debug_mode():
            import json
            print(f"Total transactions extracted: {len(transactions)}")
            with open(os.path.join(temp_output, "extracted_transactions.json"), "w") as f:
                json.dump(transactions, f, indent=4)
                
        return transactions
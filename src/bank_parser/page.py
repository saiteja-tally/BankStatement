class Page:

    def __init__(self, page_number: int):
        # Prefer the neutral name `plumber_page` (works for pdfplumber, and we alias for PyMuPDF)
        self.page_number = page_number
        self.table_ocr_data = None
        self.table_xyxy = None
        self.table_type = None
        self.table_v_lines = None
        self.table_h_lines = None
        self.table_headers = None
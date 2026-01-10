from ..dto.pipeline_dto import ExtractedText, ConfidenceScore

class OCREngine:
    def __init__(self):
        pass

    def extract_table_ocr_data(self, plumber_page, table_xyxy):
        
        # Crop the page to the table area
        x_min, y_min, x_max, y_max = table_xyxy
        cropped_page = plumber_page.within_bbox((x_min * plumber_page.width, y_min * plumber_page.height, x_max * plumber_page.width, y_max*plumber_page.height))

        # Extract OCR data from the cropped area
        ocr_data = cropped_page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)

        ocr_output = []
        for word in ocr_data:

            ocr_output.append(ExtractedText(
                text=word.get("text", ""),
                bbox_xyxy=[
                    word.get("x0", 0)/ plumber_page.width,
                    word.get("top", 0)/plumber_page.height,
                    word.get("x1", 0)/ plumber_page.width,
                    word.get("bottom", 0)/plumber_page.height
                ],
                page_no=plumber_page.page_number,
                confidence=ConfidenceScore(table_extraction=0.0, ocr=1.0, mapping=0.0, validation=0.0)
            ))

        return ocr_output


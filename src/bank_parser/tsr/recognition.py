import torch
from sentence_transformers import SentenceTransformer
from paddleocr import TableRecognitionPipelineV2
import os
import yaml
import paddle
import numpy as np

from ..config import HEADER_KEYWORDS, COMMON_HEADERS


class TableRecognizer():
    def __init__(self):
        self.transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.header_keyword_embeddings = self.transformer_model.encode(HEADER_KEYWORDS, convert_to_tensor=True, normalize_embeddings=True)
        with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
            self.tsr_config = yaml.safe_load(f)
            self.TSRecognizer = TableRecognitionPipelineV2(**self.tsr_config['paddleocrdetector'])

    def recognize_table_headers(self, page):

        """Recognize table headers using semantic similarity"""

        for i in range(len(page.table_h_lines) - 1):
            y_top = page.table_h_lines[i][1]
            y_bottom = page.table_h_lines[i + 1][1]

            row_texts = []
            for j in range(len(page.table_v_lines) - 1):
                x_left = page.table_v_lines[j][0]
                x_right = page.table_v_lines[j + 1][0]
            
                cell_texts = []
                for text_obj in page.table_ocr_data:
                    x0, y0, x1, y1 = text_obj.bbox_xyxy
                    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                    if x_left <= xc <= x_right and y_top <= yc <= y_bottom:
                        cell_texts.append(text_obj.text)

                cell_text = " ".join(cell_texts).strip()
                row_texts.append(cell_text)
            header_score = self.compute_semantic_header_score(row_texts)

            if header_score >= 0.8 and all(text.strip() for text in row_texts):
                page.table_h_lines = page.table_h_lines[i+1:]
                return row_texts

        return []
            

    def compute_semantic_header_score(self, phrases):
        """Compute semantic similarity score for header detection"""
        ## Phrases can be a list of dicts or list of strings
        if not phrases:
            return 0.0
        if isinstance(phrases[0], dict):
            texts = [p["text"].strip() for p in phrases if p["text"].strip()]
        else:
            texts = [p.strip() for p in phrases if p.strip()]
        if texts and all(t.upper() in COMMON_HEADERS for t in texts):
            return 0.0
        if not texts:
            return 0.0

        # Encode and normalize once
        block_embeddings = self.transformer_model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        # Dot product since both sets are normalized this will be equivalent to cosine similarity
        scores = torch.mm(block_embeddings, self.header_keyword_embeddings.T)

        # For each phrase, take its best match among header keywords
        max_scores, _ = torch.max(scores, dim=1)
        return torch.mean(max_scores).item() if len(max_scores) > 0 else 0.0

        
    def extract_transactions_from_page(self, page):
        
        transactions = []
        for i in range(len(page.table_h_lines) - 1):
            y_top = page.table_h_lines[i][1]
            y_bottom = page.table_h_lines[i + 1][1]

            row_texts = {}
            for j in range(len(page.table_v_lines) - 1):
                x_left = page.table_v_lines[j][0]
                x_right = page.table_v_lines[j + 1][0]
            
                cell_texts = []
                for text_obj in page.table_ocr_data:
                    x0, y0, x1, y1 = text_obj.bbox_xyxy
                    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                    if x_left <= xc <= x_right and y_top <= yc <= y_bottom:
                        cell_texts.append(text_obj.text)

                cell_text = " ".join(cell_texts).strip()
                row_texts[page.table_headers[j]] = cell_text
            
            transactions.append(row_texts)

        return transactions


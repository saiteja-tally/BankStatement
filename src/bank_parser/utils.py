import fitz  # PyMuPDF
from typing import Optional
import pdfplumber
import re,os,tempfile

from .exceptions import PDFNotReadable

def is_pdf_scanned(path: str,
                   text_threshold: int = 20,
                   page_ratio: float = 0.6,
                   password: str = None):
    """
    Returns (is_scanned, scanned_ratio, scanned_pages, total_pages).
    A page is considered 'scanned' if it has < text_threshold chars and contains >=1 image.
    The PDF is considered scanned if scanned_ratio >= page_ratio.
    """

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise PDFNotReadable(f"Unable to read PDF file.")

    
    if doc.needs_pass:
        if not password:
            return False #password is required
        if not doc.authenticate(password):
            return False #password is incorrect

    scanned_pages = 0
    total = len(doc)
    for page in doc:
        txt = page.get_text("text") or ""
        # imgs = page.get_images(full=True)
        if len(txt.strip()) < text_threshold:
            scanned_pages += 1
    ratio = scanned_pages / total if total else 0.0
    return (ratio >= page_ratio)

def is_garbage_text(text):
    # if mostly (cid:xx) patterns or non-ASCII, mark as garbage
    cid_ratio = len(re.findall(r'\(cid:\d+\)', text)) / (len(text) + 1)
    return cid_ratio > 0.1 or len(text.strip()) < 100

def normalize_pdf_to_a4_pdfplumber(src_pdf_path: str, output_path: Optional[str] = None, dpi: int = 300, password: Optional[str] = None) -> str:
    """
    Create a normalized copy of the PDF where each page is rendered and placed
    on an A4-sized page using PDFPlumber. This renders each source page to
    an image and then embeds that image into a new A4 PDF page.

    Args:
        src_pdf_path: Path to the source PDF.
        output_path: Optional path for the normalized PDF. If not provided,
                    a temporary file will be created in the system temp dir.
        dpi: Resolution for rendering PDF pages (default 300).
        password: Optional password for encrypted PDFs.

    Returns:
        Path to the normalized PDF file.
    """
    if output_path is None:
        fd, out = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        output_path = out

    # Calculate dimensions based on A4 at the specified DPI
    # A4: 210mm x 297mm = 8.27in x 11.69in
    a4_width_px = int(8.27 * dpi)
    a4_height_px = int(11.69 * dpi)

    # Open source PDF with PDFPlumber
    with pdfplumber.open(src_pdf_path, password=password) as pdf:
        # Create a new PDF to store normalized pages
        from PIL import Image
        normalized_pages = []

        for page in pdf.pages:
            # Convert page to image
            img = page.to_image(resolution=dpi)
            # Get the rendered PIL Image
            pil_image = img.original

            # Create a new A4-sized white background
            a4_image = Image.new('RGB', (a4_width_px, a4_height_px), 'white')

            # Calculate scaling to fit the page content into A4 while preserving aspect ratio
            content_ratio = pil_image.width / pil_image.height
            a4_ratio = a4_width_px / a4_height_px

            if content_ratio > a4_ratio:
                # Width is the limiting factor
                new_width = a4_width_px
                new_height = int(new_width / content_ratio)
            else:
                # Height is the limiting factor
                new_height = a4_height_px
                new_width = int(new_height * content_ratio)

            # Resize the page content
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Calculate position to center the content
            x_offset = (a4_width_px - new_width) // 2
            y_offset = (a4_height_px - new_height) // 2

            # Paste the resized content onto the A4 background
            a4_image.paste(resized_image, (x_offset, y_offset))
            normalized_pages.append(a4_image)

        # Save all pages to the output PDF
        if normalized_pages:
            normalized_pages[0].save(
                output_path,
                "PDF",
                resolution=dpi,
                save_all=True,
                append_images=normalized_pages[1:],
                quality=95
            )

    return output_path

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from ..table_detect.table_classification import TableGridType

@dataclass
class TableResult:
    """Data Transfer Object for table extraction results"""
    np_table: Optional[np.ndarray]
    method: str
    confidence: float
    hor_lines: List[List[int]]
    ver_lines: List[List[int]]
    table_type: TableGridType
    table_xyxy_wrt_plumberpage: Optional[List[int]] = None

@dataclass
class PipelineTask:
    """Data Transfer Object for Pipeline tasks"""
    page_idx: int
    ext_method: str
    image: bytes = None
    ocr_data: Dict[str, Any] = None
    hor_lines: List[List[int]] = field(default_factory=list)
    ver_lines: List[List[int]] = field(default_factory=list)
    table_type: TableGridType = TableGridType.UNBORDERED

@dataclass
class PipelineResult:
    """Data Transfer Object for Pipeline results"""
    page_idx: int
    ext_method: str
    ocr_data: Dict[str, Any]
    hor_lines: List[List[int]] = field(default_factory=list)
    ver_lines: List[List[int]] = field(default_factory=list)
    table_type: TableGridType = TableGridType.UNBORDERED

@dataclass
class PipelineConfiguration:
    """Data Transfer Object for pipeline configuration"""
    gpu_id: str = "0"
    raster_dpi: int = 300
    jpeg_quality: int = 100
    queue_maxsize: int = 4
    batch_size: int = 4
    batch_timeout: float = 0.25
    pages_to_process: List[int] = None

    def __post_init__(self):
        if self.pages_to_process is None:
            self.pages_to_process = []

@dataclass
class ProcessedPageResult:
    """Data Transfer Object for processed page results"""
    page_idx: int
    ocr_json_path: str
    structured_data: Dict[str, Any]
    table_result: TableResult

@dataclass
class ConfidenceScore:
    """Data Transfer Object for confidence scores"""
    table_extraction: float
    ocr: float
    mapping: float
    validation: float

@dataclass
class ExtractedText:
    """Data Transfer Object for the result from extraction layer"""
    text: str
    bbox_xyxy: List[float]
    page_no: int
    confidence: ConfidenceScore

    def __post_init__(self):
        if isinstance(self.confidence, dict):
            self.confidence = ConfidenceScore(**self.confidence)

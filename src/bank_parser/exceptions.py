"""
Project-wide custom exceptions.
"""
from enum import Enum


class ProcessingRetriesExceeded(Exception):
    """Raised when a file exceeds the allowed processing retry attempts."""

class InvalidFileIdError(Exception):
    """Raised when an SQS message references a missing or malformed file_id."""

class InvalidFileStatusError(Exception):
    """Raised when an SQS message references a file in a terminal status."""


class ErrorCode(Enum):
    UNKNOWN = "TAI-INF-00000"
    PROCESSING_RETRIES_EXCEEDED = "TAI-INF-01001"
    EMPTY_TRANSACTIONS_LIST = "TAI-INF-01002"
    TABLE_DETECTION_FAILED = "TAI-INF-01006"
    TABLE_STRUCTURE_RECOGNITION_FAILED = "TAI-INF-01007"
    OCR_FAILED = "TAI-INF-01008"
    MAPPER_FAILED = "TAI-INF-01004"
    VALIDATION_FAILED = "TAI-INF-01003"
    SCANNED_PDF_NOT_SUPPORTED = "TAI-INF-01009"
    PDF_NOT_READABLE = "TAI-INF-01010"
    ROW_EXTRACTION_FAILED = "TAI-INF-01011"
    REVIEW_REQUIRED_CRITICAL = "TAI-INF-01012"


class FileProcessingError(Exception):
    error_code = ErrorCode.UNKNOWN

class ValidationFailed(FileProcessingError):
    error_code = ErrorCode.VALIDATION_FAILED

class MapperFailed(FileProcessingError):
    error_code = ErrorCode.MAPPER_FAILED

class EmptyTransactionsList(FileProcessingError):
    error_code = ErrorCode.EMPTY_TRANSACTIONS_LIST

class TableDetectionFailed(FileProcessingError):
    error_code = ErrorCode.TABLE_DETECTION_FAILED

class TableStructureRecognitionFailed(FileProcessingError):
    error_code = ErrorCode.TABLE_STRUCTURE_RECOGNITION_FAILED

class OCRFailed(FileProcessingError):
    error_code = ErrorCode.OCR_FAILED

class ScannedPDFNotSupported(FileProcessingError):
    error_code = ErrorCode.SCANNED_PDF_NOT_SUPPORTED

class PDFNotReadable(FileProcessingError):
    error_code = ErrorCode.PDF_NOT_READABLE

class RowExtractionFailed(FileProcessingError):
    error_code = ErrorCode.ROW_EXTRACTION_FAILED

class ReviewRequiredCritical(FileProcessingError):
    error_code = ErrorCode.REVIEW_REQUIRED_CRITICAL

"""
Utility service with all original utils functionality
"""
from app.core.utils import *

# Re-export all utilities for backwards compatibility
__all__ = [
    "format_file_size", 
    "format_response_time",
    "truncate_text",
    "clean_text", 
    "extract_keywords",
    "create_document_summary",
    "export_chat_history",
    "validate_file_type",
    "calculate_similarity_score", 
    "format_metadata",
    "QueryHistory",
    "get_file_hash",
    "save_json",
    "load_json"
]

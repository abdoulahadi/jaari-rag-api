"""
Utility functions for the RAG application
"""
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Generate MD5 hash of a file"""
    file_path = Path(file_path)
    hash_md5 = hashlib.md5()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {str(e)}")
        return ""


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Save data to JSON file"""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None


def format_response_time(seconds: float) -> str:
    """Format response time in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10, return_frequencies: bool = False) -> Union[List[str], List[Tuple[str, int]]]:
    """Extract keywords from text (simple implementation)"""
    import re
    from collections import Counter
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words (French)
    stop_words = {
        'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
        'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
        'par', 'grand', 'fin', 'donner', 'ou', 'si', 'deux', 'très', 'faire', 'bien'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count and return most common
    counter = Counter(filtered_words)
    most_common = counter.most_common(max_keywords)
    
    if return_frequencies:
        return most_common  # List of (word, frequency) tuples
    else:
        return [word for word, count in most_common]  # List of words only


def create_document_summary(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for documents"""
    if not documents:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "average_chunk_size": 0,
            "file_types": {},
            "sources": []
        }
    
    total_chunks = len(documents)
    total_chars = sum(len(doc.get('content', '')) for doc in documents)
    
    # Count file types
    file_types = {}
    sources = set()
    
    for doc in documents:
        metadata = doc.get('metadata', {})
        file_type = metadata.get('file_type', 'unknown')
        source = metadata.get('source', 'unknown')
        
        file_types[file_type] = file_types.get(file_type, 0) + 1
        sources.add(source)
    
    return {
        "total_documents": len(sources),
        "total_chunks": total_chunks,
        "average_chunk_size": total_chars // total_chunks if total_chunks > 0 else 0,
        "total_characters": total_chars,
        "file_types": file_types,
        "sources": list(sources)
    }


def export_chat_history(messages: List[Dict[str, Any]], file_path: Union[str, Path]) -> bool:
    """Export chat history to file"""
    try:
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            return save_json({"messages": messages, "exported_at": datetime.now().isoformat()}, file_path)
        
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Chat History - Exported at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, msg in enumerate(messages, 1):
                    f.write(f"Message {i}:\n")
                    f.write(f"Role: {msg.get('role', 'unknown')}\n")
                    f.write(f"Content: {msg.get('content', '')}\n")
                    f.write(f"Timestamp: {msg.get('timestamp', 'unknown')}\n")
                    f.write("-" * 30 + "\n\n")
            
            return True
        
        else:
            logger.error(f"Unsupported export format: {file_path.suffix}")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting chat history: {str(e)}")
        return False


def validate_file_type(file_name: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is allowed"""
    file_ext = Path(file_name).suffix.lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]


def calculate_similarity_score(query: str, text: str) -> float:
    """Calculate simple similarity score between query and text"""
    from difflib import SequenceMatcher
    
    # Simple similarity calculation
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for display"""
    formatted = []
    
    for key, value in metadata.items():
        if key == 'source':
            formatted.append(f"📄 Source: {value}")
        elif key == 'file_type':
            formatted.append(f"📝 Type: {value}")
        elif key == 'page':
            formatted.append(f"📃 Page: {value}")
        elif key == 'upload_source':
            formatted.append(f"📤 Uploaded: {'Yes' if value else 'No'}")
        else:
            formatted.append(f"{key}: {value}")
    
    return " | ".join(formatted)


class QueryHistory:
    """Simple query history management"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history = []
    
    def add_query(self, query: str, response: str, response_time: float, sources: Optional[List[Dict]] = None):
        """Add query to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "response_time": response_time,
            "sources": sources or []
        }
        
        self.history.insert(0, entry)
        
        # Keep only max_history entries
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent queries"""
        return self.history[:limit]
    
    def search_history(self, search_term: str) -> List[Dict[str, Any]]:
        """Search in query history"""
        results = []
        search_term = search_term.lower()
        
        for entry in self.history:
            if (search_term in entry["query"].lower() or 
                search_term in entry["response"].lower()):
                results.append(entry)
        
        return results
    
    def clear_history(self):
        """Clear query history"""
        self.history = []
    
    def export_history(self, file_path: Union[str, Path]) -> bool:
        """Export history to file"""
        return export_chat_history(self.history, file_path)

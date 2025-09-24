"""
Document loader module with enhanced PDF support for multiple file formats
"""
import os
import sys
import tempfile
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from contextlib import contextmanager
from io import StringIO
import logging

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader, 
    DirectoryLoader, 
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_ENHANCED = True
except ImportError:
    PDF_ENHANCED = False

# OCR imports for scanned PDFs
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@contextmanager
def suppress_pdf_warnings():
    """Context manager to suppress PDF parsing warnings"""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


class EnhancedPDFLoader:
    """Enhanced PDF loader with better text extraction and metadata"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """Load PDF with enhanced text extraction and OCR fallback"""
        documents = []
        
        try:
            # Try pdfplumber first for better text extraction
            if PDF_ENHANCED:
                documents = self._load_with_pdfplumber()
                if documents and self._has_meaningful_text(documents):
                    logger.info(f"PDF loaded successfully with pdfplumber: {self.file_path}")
                    return documents
            
            # Fallback to PyPDFLoader
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            
            # Check if we got meaningful text
            if not self._has_meaningful_text(documents):
                logger.warning(f"PDF appears to be scanned (no extractable text): {self.file_path}")
                
                # Try OCR if available
                if OCR_AVAILABLE:
                    logger.info(f"Attempting OCR extraction for: {self.file_path}")
                    ocr_documents = self._load_with_ocr()
                    if ocr_documents and self._has_meaningful_text(ocr_documents):
                        logger.info(f"OCR extraction successful: {self.file_path}")
                        return ocr_documents
                
                # If OCR not available or failed, log warning and return empty text documents
                logger.warning(f"Cannot extract text from scanned PDF (OCR not available): {self.file_path}")
                # Still return documents with metadata for tracking
                for doc in documents:
                    doc.metadata.update({
                        "extraction_status": "failed_scanned_pdf",
                        "requires_ocr": True,
                        "ocr_available": OCR_AVAILABLE
                    })
            
            # Clean and enhance the extracted text
            for doc in documents:
                doc.page_content = self._clean_pdf_text(doc.page_content)
                doc.metadata.update(self._extract_pdf_metadata(doc))
            
            logger.info(f"PDF loaded with PyPDFLoader: {self.file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {self.file_path}: {str(e)}")
            return []
    
    def _load_with_pdfplumber(self) -> List[Document]:
        """Load PDF using pdfplumber for better text extraction"""
        try:
            import pdfplumber
            import warnings
            
            documents = []
            
            # Suppress PDF parsing warnings and stderr messages
            with warnings.catch_warnings(), suppress_pdf_warnings():
                warnings.filterwarnings("ignore", message=".*Cannot set gray non-stroke color.*")
                warnings.filterwarnings("ignore", message=".*is an invalid float value.*")
                
                with pdfplumber.open(self.file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text with layout preservation
                        text = page.extract_text()
                        
                        if text and text.strip():
                            # Clean the text
                            cleaned_text = self._clean_pdf_text(text)
                            
                            # Create document with enhanced metadata
                            doc = Document(
                                page_content=cleaned_text,
                                metadata={
                                    "source": self.file_path,
                                    "page": page_num + 1,
                                    "total_pages": len(pdf.pages),
                                    "extraction_method": "pdfplumber",
                                    "file_type": ".pdf",
                                    "page_width": page.width,
                                    "page_height": page.height,
                                    "char_count": len(cleaned_text),
                                    "word_count": len(cleaned_text.split())
                                }
                            )
                            documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return []
    
    def _load_with_ocr(self) -> List[Document]:
        """Load PDF using OCR for scanned documents"""
        if not OCR_AVAILABLE:
            logger.warning("OCR libraries not available")
            return []
        
        try:
            # Convert PDF to images
            images = convert_from_path(self.file_path, dpi=200)
            documents = []
            
            for page_num, image in enumerate(images):
                try:
                    # Perform OCR on the image
                    # Use French + English for agricultural documents
                    text = pytesseract.image_to_string(
                        image, 
                        lang='fra+eng',
                        config='--psm 6'  # Assume a single uniform block of text
                    )
                    
                    if text and text.strip():
                        # Clean the OCR text
                        cleaned_text = self._clean_ocr_text(text)
                        
                        # Create document with OCR metadata
                        doc = Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": self.file_path,
                                "page": page_num + 1,
                                "total_pages": len(images),
                                "extraction_method": "ocr",
                                "file_type": ".pdf",
                                "char_count": len(cleaned_text),
                                "word_count": len(cleaned_text.split()),
                                "ocr_confidence": "medium",  # Could be enhanced with actual confidence
                                "languages": "fra+eng"
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                    continue
            
            logger.info(f"OCR extracted text from {len(documents)} pages")
            return documents
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return []
    
    def _has_meaningful_text(self, documents: List[Document]) -> bool:
        """Check if documents contain meaningful extractable text"""
        if not documents:
            return False
        
        total_chars = sum(len(doc.page_content.strip()) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Consider meaningful if we have at least 50 characters and 10 words
        return total_chars >= 50 and total_words >= 10
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR-extracted text"""
        if not text:
            return ""
        
        # Basic OCR cleanup
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common OCR mistakes
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)  # Missing space after punctuation
        
        # Remove likely OCR artifacts
        text = re.sub(r'[^\w\s\-.,!?():;"\'À-ÿ]', '', text)  # Keep French accents
        text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single characters (likely OCR errors)
        
        return text.strip()
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean and normalize PDF extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple line breaks to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n ', '\n', text)  # Remove space after line break
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing space between words
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words across lines
        text = re.sub(r'([.!?])\n([A-Z])', r'\1 \2', text)  # Fix sentence breaks
        
        # Remove page headers/footers patterns (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Single numbers on lines (page numbers)
        text = re.sub(r'\n[A-Z\s]+\n', '\n', text)  # All caps headers
        
        return text.strip()
    
    def _extract_pdf_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract additional metadata from PDF"""
        metadata = {}
        
        try:
            # Analyze text content
            content = doc.page_content
            
            # Basic statistics
            metadata.update({
                "char_count": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.split('\n')),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
            })
            
            # Language detection (simple heuristic)
            french_words = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'dans', 'sur', 'avec']
            english_words = ['the', 'and', 'or', 'in', 'on', 'with', 'for', 'to', 'of', 'at']
            
            content_lower = content.lower()
            french_count = sum(1 for word in french_words if word in content_lower)
            english_count = sum(1 for word in english_words if word in content_lower)
            
            if french_count > english_count:
                metadata["likely_language"] = "french"
            elif english_count > french_count:
                metadata["likely_language"] = "english"
            else:
                metadata["likely_language"] = "unknown"
            
            # Content type detection
            if any(word in content_lower for word in ['culture', 'agriculture', 'plante', 'semis', 'récolte']):
                metadata["content_category"] = "agriculture"
            elif any(word in content_lower for word in ['technique', 'méthode', 'procédure']):
                metadata["content_category"] = "technical"
            else:
                metadata["content_category"] = "general"
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
        
        return metadata


class DocumentLoader:
    """Enhanced document loader supporting multiple formats"""
    
    def __init__(self):
        self.supported_extensions = [".txt", ".pdf", ".docx", ".md"]
        self.encoding = "utf-8"
        
    def load_from_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """Load all supported documents from a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        documents = []
        
        try:
            # Load different file types
            for extension in self.supported_extensions:
                if extension == ".txt":
                    docs = self._load_text_files(directory_path)
                elif extension == ".pdf":
                    docs = self._load_pdf_files(directory_path)
                elif extension == ".docx":
                    docs = self._load_docx_files(directory_path)
                elif extension == ".md":
                    docs = self._load_markdown_files(directory_path)
                else:
                    continue
                    
                documents.extend(docs)
                
        except Exception as e:
            logger.error(f"Error loading documents from {directory_path}: {str(e)}")
            
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return []
        
        extension = file_path.suffix.lower()
        documents = []
        
        try:
            if extension == ".txt":
                loader = TextLoader(str(file_path), encoding=self.encoding)
            elif extension == ".pdf":
                # Use enhanced PDF loader
                loader = EnhancedPDFLoader(str(file_path))
            elif extension == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return []
            
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": extension,
                    "file_size": file_path.stat().st_size,
                    "file_modified": file_path.stat().st_mtime
                })
            
            logger.info(f"Successfully loaded document: {file_path.name} ({len(documents)} chunks)")
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
        
        return documents

    def load_from_uploaded_files(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files (FastAPI UploadFile)"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=Path(uploaded_file.filename).suffix,
                    mode='wb'
                ) as tmp_file:
                    content = uploaded_file.file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                # Load based on file type
                extension = Path(uploaded_file.filename).suffix.lower()
                
                if extension == ".txt":
                    loader = TextLoader(tmp_path, encoding=self.encoding)
                elif extension == ".pdf":
                    # Use enhanced PDF loader
                    loader = EnhancedPDFLoader(tmp_path)
                elif extension == ".docx":
                    loader = Docx2txtLoader(tmp_path)
                elif extension == ".md":
                    loader = UnstructuredMarkdownLoader(tmp_path)
                else:
                    logger.warning(f"Unsupported file type: {extension}")
                    continue
                
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": uploaded_file.filename,
                        "file_type": extension,
                        "upload_source": True
                    })
                
                documents.extend(docs)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                logger.error(f"Error loading {uploaded_file.filename}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} documents from uploaded files")
        return documents
    
    def _load_text_files(self, directory_path: Path) -> List[Document]:
        """Load text files from directory"""
        try:
            loader = DirectoryLoader(
                str(directory_path),
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": self.encoding}
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading text files: {str(e)}")
            return []
    
    def _load_pdf_files(self, directory_path: Path) -> List[Document]:
        """Load PDF files from directory with enhanced processing"""
        documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                # Use enhanced PDF loader
                loader = EnhancedPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Add file-level metadata
                for doc in docs:
                    doc.metadata.update({
                        "file_name": pdf_file.name,
                        "file_size": pdf_file.stat().st_size,
                        "file_modified": pdf_file.stat().st_mtime
                    })
                
                documents.extend(docs)
                logger.info(f"Successfully loaded PDF: {pdf_file.name} ({len(docs)} pages)")
                
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file}: {str(e)}")
        
        return documents
    
    def _load_docx_files(self, directory_path: Path) -> List[Document]:
        """Load DOCX files from directory"""
        documents = []
        docx_files = list(directory_path.glob("*.docx"))
        
        for docx_file in docx_files:
            try:
                loader = Docx2txtLoader(str(docx_file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading DOCX {docx_file}: {str(e)}")
        
        return documents
    
    def _load_markdown_files(self, directory_path: Path) -> List[Document]:
        """Load Markdown files from directory"""
        documents = []
        md_files = list(directory_path.glob("*.md"))
        
        for md_file in md_files:
            try:
                loader = UnstructuredMarkdownLoader(str(md_file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading Markdown {md_file}: {str(e)}")
        
        return documents


class DocumentSplitter:
    """Document splitter with intelligent chunking for technical documents"""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or 2000  # Augmenté pour les docs techniques
        self.chunk_overlap = chunk_overlap or 400  # Plus d'overlap pour garder le contexte
        
        # Séparateurs optimisés pour documents techniques agricoles
        self.technical_separators = [
            "\n\n\n",  # Sections majeures
            "\n\n",    # Paragraphes
            "\n## ",   # Titres de niveau 2
            "\n# ",    # Titres principaux
            "\n- ",    # Listes à puces
            "\n• ",    # Puces alternatives
            "\n1. ",   # Listes numérotées
            "\n2. ",   # Continuation listes
            "\n3. ",
            ". ",      # Fin de phrases
            "\n",      # Retours à la ligne
            " ",       # Espaces
            ""         # Caractères
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.technical_separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with enhanced processing for technical content"""
        try:
            enhanced_chunks = []
            
            for doc in documents:
                # Pré-traitement pour les documents TPS
                if self._is_technical_document(doc):
                    chunks = self._smart_split_technical_document(doc)
                else:
                    chunks = self.splitter.split_documents([doc])
                
                # Post-traitement des chunks
                for chunk in chunks:
                    enhanced_chunk = self._enhance_chunk_metadata(chunk, doc)
                    enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"Split {len(documents)} documents into {len(enhanced_chunks)} enhanced chunks")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            return []
    
    def _is_technical_document(self, doc: Document) -> bool:
        """Detect if document is a technical TPS document"""
        source = doc.metadata.get('source', '').upper()
        content = doc.page_content.upper()
        
        technical_indicators = [
            'TPS', 'TECHNIQUE', 'PRODUCTION', 'SEMENCE', 'CULTURE',
            'AGRICULTURAL', 'FARMING', 'CROP', 'CULTIVATION'
        ]
        
        return any(indicator in source or indicator in content for indicator in technical_indicators)
    
    def _smart_split_technical_document(self, doc: Document) -> List[Document]:
        """Smart splitting for technical documents preserving logical structure"""
        content = doc.page_content
        source_file = doc.metadata.get('source', '')
        page_num = doc.metadata.get('page', 1)
        
        # Identifier les sections techniques
        sections = self._identify_technical_sections(content)
        
        if not sections:
            # Fallback au splitting classique
            return self.splitter.split_documents([doc])
        
        chunks = []
        for section in sections:
            # Créer des chunks pour chaque section technique
            section_chunks = self._create_section_chunks(section, doc.metadata)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _identify_technical_sections(self, content: str) -> List[Dict[str, Any]]:
        """Identify technical sections in agricultural documents"""
        import re
        
        sections = []
        
        # Patterns pour identifier les sections techniques
        section_patterns = [
            # Sections principales
            (r'(?:^|\n)\s*(\d+\.?\s*[A-Z][^.\n]{10,100})', 'main_section'),
            # Étapes de production
            (r'(?:^|\n)\s*([A-Z][^.\n]*(?:PRODUCTION|CULTURE|SEMIS|PLANTATION|RÉCOLTE)[^.\n]*)', 'production_step'),
            # Recommandations techniques
            (r'(?:^|\n)\s*([A-Z][^.\n]*(?:RECOMMANDATION|CONSEIL|TECHNIQUE)[^.\n]*)', 'recommendation'),
            # Informations générales
            (r'(?:^|\n)\s*([A-Z][^.\n]*(?:INFORMATION|GÉNÉRAL|CARACTÉRISTIQUE)[^.\n]*)', 'general_info'),
        ]
        
        for pattern, section_type in section_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                start_pos = match.start()
                title = match.group(1).strip()
                
                # Trouver la fin de la section
                next_match = None
                for next_pattern, _ in section_patterns:
                    next_matches = list(re.finditer(next_pattern, content[start_pos + len(title):], re.MULTILINE | re.IGNORECASE))
                    if next_matches:
                        if next_match is None or next_matches[0].start() < next_match:
                            next_match = next_matches[0].start() + start_pos + len(title)
                
                end_pos = next_match if next_match else len(content)
                section_content = content[start_pos:end_pos].strip()
                
                if len(section_content) > 50:  # Ignorer les sections trop courtes
                    sections.append({
                        'title': title,
                        'content': section_content,
                        'type': section_type,
                        'start': start_pos,
                        'end': end_pos
                    })
        
        # Si aucune section identifiée, créer des sections par paragraphe
        if not sections:
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 100:  # Paragraphes significatifs
                    sections.append({
                        'title': f'Section {i+1}',
                        'content': paragraph.strip(),
                        'type': 'paragraph',
                        'start': 0,
                        'end': len(paragraph)
                    })
        
        return sections
    
    def _create_section_chunks(self, section: Dict[str, Any], base_metadata: Dict[str, Any]) -> List[Document]:
        """Create optimized chunks from a technical section"""
        content = section['content']
        
        # Pour les sections courtes, garder entières
        if len(content) <= self.chunk_size:
            chunk_metadata = {
                **base_metadata,
                'section_title': section['title'],
                'section_type': section['type'],
                'chunk_index': 0,
                'total_chunks': 1
            }
            return [Document(page_content=content, metadata=chunk_metadata)]
        
        # Pour les sections longues, découper intelligemment
        chunks = []
        
        # Découper par sous-sections ou paragraphes
        sub_parts = content.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for part in sub_parts:
            # Si ajouter cette partie dépasse la taille, créer un nouveau chunk
            if len(current_chunk) + len(part) > self.chunk_size and current_chunk:
                # Ajouter contexte de la section
                enhanced_content = f"[{section['title']}]\n\n{current_chunk}"
                
                chunk_metadata = {
                    **base_metadata,
                    'section_title': section['title'],
                    'section_type': section['type'],
                    'chunk_index': chunk_index,
                    'enhanced_content': True
                }
                
                chunks.append(Document(page_content=enhanced_content, metadata=chunk_metadata))
                
                # Commencer nouveau chunk avec overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + part
                chunk_index += 1
            else:
                current_chunk += "\n\n" + part if current_chunk else part
        
        # Ajouter le dernier chunk
        if current_chunk:
            enhanced_content = f"[{section['title']}]\n\n{current_chunk}"
            chunk_metadata = {
                **base_metadata,
                'section_title': section['title'],
                'section_type': section['type'],
                'chunk_index': chunk_index,
                'enhanced_content': True
            }
            chunks.append(Document(page_content=enhanced_content, metadata=chunk_metadata))
        
        # Mettre à jour total_chunks
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _enhance_chunk_metadata(self, chunk: Document, original_doc: Document) -> Document:
        """Enhance chunk metadata with additional context"""
        # Analyser le contenu pour des métadonnées additionnelles
        content = chunk.page_content.lower()
        
        # Détection de mots-clés agricoles
        agricultural_keywords = {
            'pomme_de_terre': ['pomme de terre', 'tubercule', 'potato'],
            'haricot': ['haricot', 'bean', 'légumineuse'],
            'laitue': ['laitue', 'lettuce', 'salade'],
            'tomate': ['tomate', 'tomato'],
            'maïs': ['maïs', 'corn', 'céréale'],
            'riz': ['riz', 'rice'],
            'manioc': ['manioc', 'cassava']
        }
        
        detected_crops = []
        for crop, keywords in agricultural_keywords.items():
            if any(keyword in content for keyword in keywords):
                detected_crops.append(crop)
        
        # Détection de types d'information
        info_types = []
        if any(word in content for word in ['semis', 'plantation', 'semer']):
            info_types.append('plantation')
        if any(word in content for word in ['irrigation', 'arrosage', 'eau']):
            info_types.append('irrigation')
        if any(word in content for word in ['fertilisant', 'engrais', 'nutriment']):
            info_types.append('fertilisation')
        if any(word in content for word in ['maladie', 'parasite', 'traitement']):
            info_types.append('protection')
        if any(word in content for word in ['récolte', 'harvest', 'cueillette']):
            info_types.append('recolte')
        
        # Enrichir les métadonnées
        enhanced_metadata = {
            **chunk.metadata,
            'detected_crops': detected_crops,
            'info_types': info_types,
            'content_length': len(chunk.page_content),
            'word_count': len(chunk.page_content.split())
        }
        
        return Document(page_content=chunk.page_content, metadata=enhanced_metadata)
    
    def update_parameters(self, chunk_size: int, chunk_overlap: int):
        """Update splitter parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

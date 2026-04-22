import asyncio
import io
import os
import tempfile
from pathlib import Path

def _extract_pdf_sync(file_bytes: bytes) -> str:
    """Synchronous function to extract PDF text using pymupdf4llm."""
    import pymupdf4llm
    
    # pymupdf4llm prefers reading from a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Convert to markdown directly
        markdown_text = pymupdf4llm.to_markdown(tmp_path)
        return markdown_text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _extract_docx_sync(file_bytes: bytes) -> str:
    """Synchronous function to extract DOCX text using python-docx."""
    from docx import Document
    
    # Process from memory buffer
    buffer = io.BytesIO(file_bytes)
    doc = Document(buffer)
    
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


async def parse_document(file_bytes: bytes, filename: str) -> str:
    """Parse a document and extract its text content gracefully.
    
    Args:
        file_bytes: The raw bytes of the uploaded file.
        filename: The original filename to deduce format (.pdf or .docx).
        
    Returns:
        The extracted markdown or raw text.
    """
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        return await asyncio.to_thread(_extract_pdf_sync, file_bytes)
    elif ext in (".docx", ".doc"):
        # For strictly .docx
        if ext == ".doc":
            raise ValueError("Legacy .doc files are not supported. Please convert to .docx.")
        return await asyncio.to_thread(_extract_docx_sync, file_bytes)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Must be .pdf or .docx.")

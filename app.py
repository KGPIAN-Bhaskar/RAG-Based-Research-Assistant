"""
Document Parsing and Text Chunking Engine for RAG Pipeline.
Extracts clean text from PDF, DOCX, TXT, and Markdown files and performs recursive text splitting.
"""

import time
from typing import List, Dict, Any, Tuple
import docx2txt
import pypdf


class SimpleTextSplitter:
    """
    Recursive Character Text Splitter with configurable chunk size and overlap.
    Splits documents along paragraph, sentence, and word boundaries.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_recursive(txt: str, seps: List[str]) -> List[str]:
            if len(txt) <= self.chunk_size:
                return [txt]
            
            if not seps:
                step = max(1, self.chunk_size - self.chunk_overlap)
                return [txt[i:i + self.chunk_size] for i in range(0, len(txt), step)]
            
            sep = seps[0]
            parts = txt.split(sep)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    if len(part) > self.chunk_size:
                        sub_chunks = split_recursive(part, seps[1:])
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        raw_chunks = split_recursive(text, separators)
        
        # Apply overlapping window to preserve boundary context across chunks
        overlapped_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if i == 0 or self.chunk_overlap <= 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = raw_chunks[i - 1]
                overlap_text = (
                    prev_chunk[-self.chunk_overlap:]
                    if len(prev_chunk) > self.chunk_overlap
                    else prev_chunk
                )
                overlapped_chunks.append(f"{overlap_text} {chunk}")
        return overlapped_chunks


def parse_document(
    uploaded_file: Any, chunk_size: int, chunk_overlap: int = 150
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Parses an uploaded file buffer (.pdf, .docx, .txt, .md) and returns text chunks,
    corresponding metadata dicts, and unique chunk IDs.
    """
    filename = uploaded_file.name
    splitter = SimpleTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    
    if filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(uploaded_file)
        pages_content: List[Tuple[str, int]] = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                pages_content.append((text, page_num + 1))
        
        for text, page_no in pages_content:
            page_chunks = splitter.split_text(text)
            for idx, ch in enumerate(page_chunks):
                chunks.append(ch)
                metadatas.append({
                    "source": filename,
                    "page": page_no,
                    "chunk_index": idx
                })
                ids.append(f"{filename}_p{page_no}_c{idx}_{time.time_ns()}")
                
    else:
        if filename.endswith(".docx"):
            file_content = docx2txt.process(uploaded_file)
        else:  # txt or md
            raw_bytes = uploaded_file.read()
            try:
                file_content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                file_content = raw_bytes.decode("latin-1", errors="ignore")
            
        raw_chunks = splitter.split_text(file_content)
        for idx, ch in enumerate(raw_chunks):
            chunks.append(ch)
            metadatas.append({
                "source": filename,
                "page": 1,
                "chunk_index": idx
            })
            ids.append(f"{filename}_c{idx}_{time.time_ns()}")
            
    return chunks, metadatas, ids

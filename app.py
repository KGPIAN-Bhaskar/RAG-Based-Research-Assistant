import time
import docx2txt
import pypdf

# SimpleTextSplitter and document parsing logic are defined below

# --- Simple & Robust Text Splitter ---
class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_recursive(txt: str, seps: list[str]) -> list[str]:
            if len(txt) <= self.chunk_size:
                return [txt]
            
            if not seps:
                return [txt[i:i+self.chunk_size] for i in range(0, len(txt), self.chunk_size - self.chunk_overlap)]
            
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

        chunks = split_recursive(text, separators)
        
        # Prepend overlap from previous chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                overlapped_chunks.append(overlap_text + " " + chunk)
        return overlapped_chunks


# --- Document Content Extractor and Parser ---
def parse_document(uploaded_file, chunk_size: int, chunk_overlap: int = 150):
    filename = uploaded_file.name
    splitter = SimpleTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunks = []
    metadatas = []
    ids = []
    
    if filename.endswith(".pdf"):
        # Read PDF
        pdf_reader = pypdf.PdfReader(uploaded_file)
        pages_content = []
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
        # Read docx or text/md
        if filename.endswith(".docx"):
            file_content = docx2txt.process(uploaded_file)
        else:  # txt or md
            file_content = uploaded_file.read().decode("utf-8")
            
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

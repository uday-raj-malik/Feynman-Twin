import os
import fitz  # PyMuPDF

def load_document_file(file_path, doc_type):
    """
    Loads a single document file (PDF or TXT) and returns a list of page objects.
    Each page object has the structure:
    {
      "text": str,
      "metadata": {
        "source": str,
        "page": int,
        "title": str,
        "document_type": str
      }
    }
    """
    pages_data = []
    source_name = os.path.basename(file_path)
    
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            title = doc.metadata.get("title") or source_name
            
            for idx, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages_data.append({
                        "text": text,
                        "metadata": {
                            "source": source_name,
                            "page": idx + 1,  # 1-indexed
                            "title": title,
                            "document_type": doc_type
                        }
                    })
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if text.strip():
                pages_data.append({
                    "text": text,
                    "metadata": {
                        "source": source_name,
                        "page": 1,
                        "title": source_name,
                        "document_type": doc_type
                    }
                })
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        
    return pages_data

def load_data_directory(data_dir):
    """
    Recursively scans the data_dir for folders (books, interviews, lectures, papers)
    and loads all PDFs and TXT files.
    """
    all_pages = []
    if not os.path.exists(data_dir):
        return all_pages
        
    subdirs = ["books", "interviews", "lectures", "papers"]
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    pages = load_document_file(file_path, subdir)
                    all_pages.extend(pages)
                    
    return all_pages

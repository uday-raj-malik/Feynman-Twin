import re

class HybridChunker:
    """
    Splits text into structural paragraphs and further into sentences.
    Computes a sentence-level sliding window context of size 3 (i-3 to i+3).
    """
    def __init__(self, window_size=3):
        self.window_size = window_size

    def split_into_sentences(self, text):
        """Splits a block of text into sentences using a regex boundary detector."""
        # Replace newlines with spaces to normalize sentence extraction
        text = re.sub(r'\s+', ' ', text).strip()
        # Regex to split sentences (avoiding common abbreviations like Mr. or i.e.)
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, document):
        """
        Chunks a single document page.
        Returns a list of chunks, each containing the original sentence,
        the full window context, and base metadata.
        """
        text = document["text"]
        metadata = document["metadata"]
        
        # 1. Structure awareness: Split into paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        
        # 2. Extract all sentences in order across paragraphs
        all_sentences = []
        for para in paragraphs:
            if para.strip():
                sentences = self.split_into_sentences(para)
                all_sentences.extend(sentences)
                
        chunks = []
        total_sentences = len(all_sentences)
        
        # 3. Generate sentence windows
        for i in range(total_sentences):
            center_sentence = all_sentences[i]
            
            # Clamp boundaries
            start_idx = max(0, i - self.window_size)
            end_idx = min(total_sentences, i + self.window_size + 1)
            
            window_context = " ".join(all_sentences[start_idx:end_idx])
            
            chunks.append({
                "original_sentence": center_sentence,
                "window": window_context,
                "sentence_index": i,
                "metadata": metadata.copy()
            })
            
        return chunks

    def chunk_all(self, documents):
        """Chunks a list of documents and flattens the resulting list."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

"""
Chunk 8-K filings into smaller segments for RAG retrieval
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    TEXTS_2019, LABELS_2019, CHUNKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
)


class Filing8KChunker:
    """Chunk 8-K filings by items and paragraphs"""
    
    # Common 8-K item patterns
    ITEM_PATTERNS = [
        r'\[ITEM (\d+\.\d+):[^\]]*\]',  # [ITEM 2.02: ...]
        r'Item (\d+\.\d+)[:\s]',         # Item 2.02: or Item 2.02
        r'ITEM (\d+\.\d+)[:\s]',         # ITEM 2.02:
    ]
    
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_items(self, text: str) -> List[Tuple[str, str]]:
        """Extract items and their content from 8-K text"""
        items = []
        
        # Find all item boundaries
        item_positions = []
        for pattern in self.ITEM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                item_positions.append((match.start(), match.group(1), match.end()))
        
        # Sort by position
        item_positions.sort(key=lambda x: x[0])
        
        # Extract content between items
        for i, (start, item_num, header_end) in enumerate(item_positions):
            # Get content until next item or end of text
            if i < len(item_positions) - 1:
                end = item_positions[i + 1][0]
            else:
                end = len(text)
            
            content = text[header_end:end].strip()
            if content:
                items.append((item_num, content))
        
        # If no items found, treat entire text as one item
        if not items:
            items.append(("FULL", text))
            
        return items
    
    def chunk_text(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size
            
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If paragraph itself is too long, split by sentences
            if len(para) > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if len(current_chunk) >= MIN_CHUNK_SIZE:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            else:
                # Add paragraph to current chunk
                if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if len(current_chunk) >= MIN_CHUNK_SIZE:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
        
        # Add remaining chunk
        if len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if self.overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk to beginning
                    prev_end = chunks[i-1][-self.overlap:]
                    chunk = prev_end + " " + chunk
                if i < len(chunks) - 1:
                    # Add beginning of next chunk to end
                    next_start = chunks[i+1][:self.overlap]
                    chunk = chunk + " " + next_start
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks
            
        return chunks
    
    def process_filing(self, filing_data: Dict) -> List[Dict]:
        """Process a single filing into chunks"""
        chunks_data = []
        
        text = filing_data.get('text', '')
        if not text:
            return chunks_data
            
        ticker = filing_data.get('ticker', '')
        filing_date = filing_data.get('filing_date', '')
        
        # Extract items from the filing
        items = self.extract_items(text)
        
        chunk_id = 0
        for item_num, item_content in items:
            # Chunk the item content
            item_chunks = self.chunk_text(item_content)
            
            for position, chunk_text in enumerate(item_chunks):
                chunk_data = {
                    'chunk_id': f"{ticker}_{filing_date}_{chunk_id}",
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'item_number': item_num,
                    'chunk_position': position,
                    'chunk_text': chunk_text,
                    'chunk_length': len(chunk_text),
                    'total_chunks_in_item': len(item_chunks)
                }
                chunks_data.append(chunk_data)
                chunk_id += 1
                
        return chunks_data


def main():
    """Process all 2019 8-K filings"""
    print("Loading 2019 8-K filings...")
    
    # Load filing texts
    with open(TEXTS_2019, 'r') as f:
        texts_data = json.load(f)
    
    # Load labels for merging
    labels_df = pd.read_csv(LABELS_2019)
    labels_dict = {}
    for _, row in labels_df.iterrows():
        labels_dict[row['accession']] = {
            'signal': row['signal'],
            'adjusted_return_pct': row['adjusted_return_pct'],
            'outperformed_market': row['outperformed_market']
        }
    
    print(f"Loaded {len(texts_data)} filings")
    
    # Initialize chunker
    chunker = Filing8KChunker()
    
    # Process all filings
    all_chunks = []
    filing_metadata = []
    
    for accession, filing_data in tqdm(texts_data.items(), desc="Chunking filings"):
        # Add accession to filing data
        filing_data['accession'] = accession
        
        # Process into chunks
        chunks = chunker.process_filing(filing_data)
        
        # Add accession and labels to each chunk
        for chunk in chunks:
            chunk['accession'] = accession
            if accession in labels_dict:
                chunk.update(labels_dict[accession])
            
        all_chunks.extend(chunks)
        
        # Save filing-level metadata
        filing_meta = {
            'accession': accession,
            'ticker': filing_data.get('ticker', ''),
            'filing_date': filing_data.get('filing_date', ''),
            'num_chunks': len(chunks),
            'total_chars': sum(c['chunk_length'] for c in chunks)
        }
        if accession in labels_dict:
            filing_meta.update(labels_dict[accession])
        filing_metadata.append(filing_meta)
    
    # Save chunks to JSON
    output_file = CHUNKS_DIR / "chunks_2019.json"
    with open(output_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Save metadata
    metadata_df = pd.DataFrame(filing_metadata)
    metadata_df.to_csv(CHUNKS_DIR / "filing_metadata_2019.csv", index=False)
    
    # Print statistics
    print(f"\nChunking complete!")
    print(f"Total filings processed: {len(texts_data)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Average chunks per filing: {len(all_chunks) / len(texts_data):.1f}")
    print(f"Average chunk size: {sum(c['chunk_length'] for c in all_chunks) / len(all_chunks):.0f} chars")
    print(f"\nChunks saved to: {output_file}")
    print(f"Metadata saved to: {CHUNKS_DIR / 'filing_metadata_2019.csv'}")
    
    # Show sample chunk
    if all_chunks:
        print("\nSample chunk:")
        sample = all_chunks[0]
        print(f"  Ticker: {sample['ticker']}")
        print(f"  Item: {sample['item_number']}")
        print(f"  Length: {sample['chunk_length']} chars")
        print(f"  Text preview: {sample['chunk_text'][:200]}...")


if __name__ == "__main__":
    main()
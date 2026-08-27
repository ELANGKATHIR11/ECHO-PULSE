import os
import json
import math
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

class SystemIRIndexer:
    """
    Project Information Retrieval (IR) and Semantic BM25 Indexing Engine
    for EchoPulseNet platform documentation, reports, metadata, configs, and codebase.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'from', 'as', 'is', 'it', 'this', 'that'}
        return [w for w in words if len(w) > 1 and w not in stopwords]

    def build_index(self, max_file_size_kb: int = 1024):
        valid_exts = {'.md', '.json', '.py', '.ts', '.tsx', '.txt', '.yml', '.yaml'}
        skip_dirs = {'node_modules', '.git', '.pytest_cache', '__pycache__', 'dist', 'data'}

        doc_id = 0
        for path in self.root_dir.rglob('*'):
            if any(part in skip_dirs for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in valid_exts:
                if path.stat().st_size > max_file_size_kb * 1024:
                    continue
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue

                rel_path = str(path.relative_to(self.root_dir))
                tokens = self.tokenize(content) + self.tokenize(rel_path) * 3
                if not tokens:
                    continue

                term_counts = Counter(tokens)
                self.documents.append({
                    "id": doc_id,
                    "rel_path": rel_path,
                    "name": path.name,
                    "ext": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "snippet": content[:300].replace('\n', ' ')
                })
                self.doc_lengths.append(len(tokens))

                for term, freq in term_counts.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = {}
                    self.inverted_index[term][doc_id] = freq

                doc_id += 1

        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.total_docs if self.total_docs > 0 else 0

        # Calculate IDF (BM25 smoothed)
        for term, posting in self.inverted_index.items():
            n_q = len(posting)
            self.idf[term] = math.log(1.0 + (self.total_docs - n_q + 0.5) / (n_q + 0.5))

    def query(self, query_str: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        tokens = self.tokenize(query_str)
        scores: Dict[int, float] = {}

        for term in tokens:
            if term not in self.inverted_index:
                continue
            idf_val = self.idf[term]
            for doc_id, freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]
                tf_score = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_len / self.avg_doc_length)))
                score = idf_val * tf_score
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in ranked:
            doc = self.documents[doc_id]
            results.append({
                "score": round(score, 4),
                "path": doc["rel_path"],
                "name": doc["name"],
                "ext": doc["ext"],
                "size_kb": round(doc["size_bytes"] / 1024, 2),
                "preview": doc["snippet"]
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents_indexed": self.total_docs,
            "unique_vocabulary_terms": len(self.inverted_index),
            "average_doc_token_length": round(self.avg_doc_length, 2),
            "root_directory": str(self.root_dir)
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ir = SystemIRIndexer(root)
    ir.build_index()
    stats = ir.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Run test search if arguments given
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "sonar bathymetry deep learning model"
    print(f"\n--- Top IR Results for Query: '{query}' ---")
    results = ir.query(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']} | {r['path']}")
        print(f"    Preview: {r['preview'][:120]}...\n")

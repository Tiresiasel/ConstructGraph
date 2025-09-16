# build_graph_blueprint_compliant.py
# Updated to fully comply with the Technical Blueprint for Academic Knowledge Graph Prototype
import os
import json
import re
import uuid
import hashlib
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
from construct_graph.config import CONFIG
from py2neo import Graph, Node, Relationship
from datetime import datetime
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import string

from typing import Optional, Any, Dict, List
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import logging
from logging.handlers import RotatingFileHandler

# --- 1. CONFIGURATION ---
RESET_DATABASE = os.getenv('RESET_DATABASE', 'false').lower() in ('true', '1', 'yes')  # Set via environment variable, defaults to False

def format_title_case(title: str) -> str:
    """
    Format a paper title to proper title case.
    Handles common academic title formatting rules:
    - Capitalize first letter of each word
    - Keep certain words lowercase (articles, prepositions, conjunctions)
    - Handle special cases like colons, hyphens, and parentheses
    """
    if not title or not isinstance(title, str):
        return title
    
    # Words that should remain lowercase (except when first word)
    lowercase_words = {
        'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it',
        'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there',
        'these', 'they', 'this', 'to', 'was', 'will', 'with', 'within', 'without'
    }
    
    # Split by spaces and process each word
    words = title.strip().split()
    if not words:
        return title
    
    formatted_words = []
    
    for i, word in enumerate(words):
        # Clean the word (remove extra punctuation but keep internal punctuation)
        clean_word = word.strip()
        if not clean_word:
            continue
            
        # Check if it's the first word or should be capitalized
        if i == 0 or clean_word.lower() not in lowercase_words:
            # Capitalize the word
            formatted_word = clean_word.capitalize()
        else:
            # Keep lowercase for articles, prepositions, etc.
            formatted_word = clean_word.lower()
        
        formatted_words.append(formatted_word)
    
    return ' '.join(formatted_words)

# Database configuration - use CONFIG instead of hardcoded values
from construct_graph.config import CONFIG

# Qdrant vector database configuration
QDRANT_THEORY_COLLECTION_NAME = "theories"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Optional reasoning/temperature controls
# OPENAI_REASONING_EFFORT accepts: "low" | "medium" | "high" (if supported by the model)
# OPENAI_TEMPERATURE controls randomness; default 0 for extraction consistency
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "high")
_temp_env = (os.getenv("OPENAI_TEMPERATURE", "").strip())
try:
    TEMPERATURE = float(_temp_env) if _temp_env != "" else None  # None means do not send the param
except ValueError:
    TEMPERATURE = None

def _safe_chat_completion(model: str, messages: list, response_format: dict):
    """Create a chat completion with graceful fallbacks for unsupported params.

    Order of attempts:
    1) reasoning + temperature (if provided)
    2) reasoning only (drop temperature)
    3) temperature only (drop reasoning)
    4) neither reasoning nor temperature
    """
    base_kwargs = {
        "model": model,
        "response_format": response_format,
        "messages": messages,
    }

    # Attempt 1: reasoning + (optional) temperature
    try:
        kwargs = dict(base_kwargs)
        kwargs["reasoning"] = {"effort": REASONING_EFFORT}
        if TEMPERATURE is not None:
            kwargs["temperature"] = TEMPERATURE
        return client.chat.completions.create(**kwargs)
    except Exception as e1:
        msg = str(e1)
        # Attempt 2: reasoning only
        try:
            kwargs = dict(base_kwargs)
            kwargs["reasoning"] = {"effort": REASONING_EFFORT}
            return client.chat.completions.create(**kwargs)
        except Exception as e2:
            # Attempt 3: temperature only (if provided)
            try:
                kwargs = dict(base_kwargs)
                if TEMPERATURE is not None:
                    kwargs["temperature"] = TEMPERATURE
                return client.chat.completions.create(**kwargs)
            except Exception as e3:
                # Attempt 4: plain
                kwargs = dict(base_kwargs)
                return client.chat.completions.create(**kwargs)

# Database reset configuration

# File tracking configuration
PROCESSED_FILES_LOG = "processed_files.json"  # Deprecated: no longer used
OUTPUT_DIR = Path(CONFIG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency configuration
MAX_CONCURRENT_REQUESTS = 5  # Safe concurrent requests for OpenAI API
REQUEST_DELAY = 0.1  # Delay between requests to avoid rate limiting

# Updated to use papers/ directory to align with existing project structure
# Default input directory is project-level data/input, overridable via env
PDF_FOLDER = Path(CONFIG.input_dir)

# --- 1.0 LOGGING CONFIGURATION ---
def setup_logging():
    """Setup a simple, storage-friendly logging system.

    Policy:
    - Only operational steps are logged (INFO).
    - No prompt, model inputs/outputs, or large payloads are written to logs.
    - Single rotating file plus concise console output.
    """
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)

    simple_formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(simple_formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        logs_dir / "build.log",
        maxBytes=2*1024*1024,  # 2MB
        backupCount=2,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(simple_formatter)
    root.addHandler(file_handler)

    root.info("Logging initialized (simple mode). No prompts or model I/O will be logged.")
    root.info(f"Logs directory: {logs_dir}")
    return root

# --- 1.1 PAPER UID HELPERS ---
def _normalize_title_for_uid(title: str) -> str:
    s = (title or '').lower()
    # remove all symbols, keep alphanumerics and spaces
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def generate_paper_uid(title: str, year) -> str:
    base = f"{_normalize_title_for_uid(title)}::{year or ''}"
    # short, stable hash id
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

# --- 1.2 VECTOR DATABASE SETUP ---
def setup_vector_database():
    """Initialize Qdrant vector database and collections with correct dimensions.
    Uses Ollama bge-m3 model which generates 1024-dim vectors.
    In RESET_DATABASE mode, drop and recreate both collections to avoid dim mismatches.
    """
    try:
        client = qdrant_client.QdrantClient(host=CONFIG.qdrant.host, port=CONFIG.qdrant.port)
        
        # Inspect existing collections
        collections = client.get_collections()
        existing = [col.name for col in collections.collections]

        # Constructs collection
        if RESET_DATABASE and CONFIG.qdrant.collection in existing:
            try:
                client.recreate_collection(
                    collection_name=CONFIG.qdrant.collection,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"Recreated Qdrant collection (reset): {CONFIG.qdrant.collection}")
            except Exception:
                # Fallback if recreate not available
                client.delete_collection(collection_name=CONFIG.qdrant.collection)
                client.create_collection(
                    collection_name=CONFIG.qdrant.collection,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"Dropped and created Qdrant collection (reset): {CONFIG.qdrant.collection}")
        elif CONFIG.qdrant.collection not in existing:
            client.create_collection(
                collection_name=CONFIG.qdrant.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {CONFIG.qdrant.collection}")
        else:
            print(f"Using existing Qdrant collection: {CONFIG.qdrant.collection}")

        # Theories collection
        if RESET_DATABASE and QDRANT_THEORY_COLLECTION_NAME in existing:
            try:
                client.recreate_collection(
                    collection_name=QDRANT_THEORY_COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"Recreated Qdrant collection (reset): {QDRANT_THEORY_COLLECTION_NAME}")
            except Exception:
                client.delete_collection(collection_name=QDRANT_THEORY_COLLECTION_NAME)
                client.create_collection(
                    collection_name=QDRANT_THEORY_COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"Dropped and created Qdrant collection (reset): {QDRANT_THEORY_COLLECTION_NAME}")
        elif QDRANT_THEORY_COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=QDRANT_THEORY_COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {QDRANT_THEORY_COLLECTION_NAME}")
        else:
            print(f"Using existing Qdrant collection: {QDRANT_THEORY_COLLECTION_NAME}")
        
        return client
    except Exception as e:
        print(f"Error setting up vector database: {e}")
        return None

def clean_construct_name(term: str) -> str:
    """Clean construct name by removing abbreviations and standardizing format."""
    if not term:
        return term
    
    # Remove content in parentheses (including the parentheses)
    # This handles cases like "organizational commitment (OC)" -> "organizational commitment"
    cleaned = re.sub(r'\s*\([^)]*\)', '', term.strip())
    
    # Also remove content in square brackets if present
    cleaned = re.sub(r'\s*\[[^\]]*\]', '', cleaned.strip())
    
    # Normalize common variations
    # Handle "X of Y" vs "Y X" patterns (e.g., "volume of attack" vs "attack volume")
    cleaned = re.sub(r'\bvolume\s+of\s+(\w+)', r'\1 volume', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\blevel\s+of\s+(\w+)', r'\1 level', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bdegree\s+of\s+(\w+)', r'\1 degree', cleaned, flags=re.IGNORECASE)
    
    # Remove common filler words that don't add semantic meaning
    filler_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = cleaned.split()
    filtered_words = [word for word in words if word.lower() not in filler_words]
    
    # Rejoin and normalize spacing
    cleaned = ' '.join(filtered_words).strip()
    
    return cleaned

def clean_theory_name(name: str) -> str:
    """Remove parenthetical/square-bracket abbreviations; preserve original casing otherwise."""
    if not name:
        return name
    cleaned = re.sub(r'\s*\([^)]*\)', '', name.strip())
    cleaned = re.sub(r'\s*\[[^\]]*\]', '', cleaned.strip())
    return cleaned

def canonical_id_from_name(name: str) -> str:
    """Generate a canonical_id (snake_case) from a display name."""
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9\s]+', '', s)
    s = re.sub(r'\s+', '_', s)
    return s

# --- 1.3 FILE TRACKING SYSTEM ---
def load_processed_files():
    """Load the list of already processed files."""
    log_file = Path(__file__).parent / PROCESSED_FILES_LOG
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        except Exception as e:
            print(f"Error loading processed files log: {e}")
            return set()
    return set()

def save_processed_files(processed_files):
    """Save the list of processed files."""
    log_file = Path(__file__).parent / PROCESSED_FILES_LOG
    try:
        data = {
            'processed_files': list(processed_files),
            'last_updated': datetime.now().isoformat(),
            'total_processed': len(processed_files)
        }
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated processed files log: {len(processed_files)} files tracked")
    except Exception as e:
        print(f"Error saving processed files log: {e}")

def get_unprocessed_files(pdf_folder, processed_files):
    """Get list of PDF files that haven't been processed yet."""
    all_pdfs = []
    
    # Method 1: Use rglob to find ALL PDFs in any nested subdirectory
    all_nested_pdfs = list(pdf_folder.rglob("*.pdf"))
    if all_nested_pdfs:
        all_pdfs.extend(all_nested_pdfs)
    
    # Method 2: Also check for PDFs directly in the papers folder (avoid duplicates)
    direct_pdfs = list(pdf_folder.glob("*.pdf"))
    for pdf in direct_pdfs:
        if pdf not in all_pdfs:
            all_pdfs.append(pdf)
    
    # Remove duplicates and sort for consistent processing order
    all_pdfs = list(set(all_pdfs))
    all_pdfs.sort()
    
    # Filter out already processed files
    unprocessed_pdfs = [pdf for pdf in all_pdfs if str(pdf) not in processed_files]
    
    return all_pdfs, unprocessed_pdfs

def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI text-embedding-3-small model."""
    from openai import OpenAI
    import os
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # OpenAI text-embedding-3-small generates 1536-dimensional vectors
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding

def generate_theory_embedding(name: str, context: str = None) -> list:
    """Generate an embedding for a theory using its name plus optional grounding context."""
    text = (name or '').strip()
    if context:
        text = f"{text} \n {context.strip()}"
    return generate_embedding(text)

# --- 2. ENHANCED LLM EXTRACTION ---

def deduplicate_constructs_in_paper(data: dict) -> dict:
    """
    Intra-paper deduplication: merge constructs that clearly refer to the same concept.

    Strategy (cost-efficient and reliable):
    - Compute sentence-transformer embeddings for each construct (use definition if available, else term)
    - Build high-precision clusters using cosine similarity threshold
    - Select a canonical term per cluster based on frequency in relationships and definition richness
    - Remap constructs, relationships, and measurements to canonical terms

    This avoids extra LLM calls and keeps precision high within a single paper.
    """
    if not data or not isinstance(data, dict):
        return data

    constructs = data.get('constructs') or []
    if not constructs:
        return data

    # Prepare items
    def _vec(x: list) -> np.ndarray:
        arr = np.array(x, dtype=float)
        # avoid zero vector division
        n = np.linalg.norm(arr)
        return arr if n == 0 else arr / n

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    normalized_terms = []
    texts_for_embed = []
    for c in constructs:
        term_raw = c.get('term') or ''
        term_norm = clean_construct_name(term_raw).strip().lower()
        normalized_terms.append(term_norm)
        text_for_vec = c.get('definition') or term_norm
        texts_for_embed.append(text_for_vec)

    # Generate vectors
    vectors = [_vec(generate_embedding(t)) for t in texts_for_embed]

    # Union-Find for clustering
    parent = list(range(len(vectors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Keep original semantic merge threshold but add AI adjudication on top to prevent false positives
    MERGE_THRESHOLD = 0.75

    # Heuristics: treat only obvious surface-form variants as the same concept
    stopwords = {"of", "the", "a", "an", "and", "for", "to", "in", "on", "by", "with"}

    def normalize_tokens(term: str) -> list:
        t = re.sub(r"[^a-z0-9\s]", " ", term.lower()).strip()
        parts = [p for p in t.split() if p and p not in stopwords]
        # naive singularization: drop trailing 's' when applicable
        norm = []
        for p in parts:
            if len(p) > 3 and p.endswith("s"):
                norm.append(p[:-1])
            else:
                norm.append(p)
        return sorted(norm)

    def are_obvious_variants(a: str, b: str) -> bool:
        if not a or not b:
            return False
        # exact after cleanup
        if a == b:
            return True
        ta = normalize_tokens(a)
        tb = normalize_tokens(b)
        if ta == tb and len(ta) > 0:
            return True
        # high token Jaccard with same multiset size implies reordering or minor inflection
        sa, sb = set(ta), set(tb)
        if sa and sb:
            jacc = len(sa & sb) / len(sa | sb)
            if jacc >= 0.9:
                return True
        return False
    
    # CRITICAL FIX: For constructs from the same paper, exact name matches should ALWAYS be merged
    # This prevents the "similar construct connection" issue you're experiencing
    def should_merge_constructs(construct_a, construct_b) -> bool:
        # If they have the same paper_uid, they're from the same paper
        paper_a = construct_a.get('paper_uid') or construct_a.get('paper_ids', [])
        paper_b = construct_b.get('paper_uid') or construct_b.get('paper_ids', [])
        
        # If both have paper information and they're from the same paper
        if paper_a and paper_b:
            # Convert to sets for comparison
            papers_a = set(paper_a) if isinstance(paper_a, list) else {paper_a}
            papers_b = set(paper_b) if isinstance(paper_b, list) else {paper_b}
            
            # If they share any papers, they're from the same source
            if papers_a & papers_b:
                # For same-paper constructs, exact name match should always merge
                return are_obvious_variants(normalized_terms[constructs.index(construct_a)], 
                                         normalized_terms[constructs.index(construct_b)])
        
        # For different papers, use the original logic
        return False
    
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            # CRITICAL FIX: Check if constructs should be merged based on paper source
            if should_merge_constructs(constructs[i], constructs[j]):
                union(i, j)
                continue
            
            # Exact or obvious surface-form variant match (after cleaning)
            if are_obvious_variants(normalized_terms[i], normalized_terms[j]):
                union(i, j)
                continue
            
            # Check for name variations (e.g., "volume of attack" vs "attack volume")
            term_i_words = set(normalized_terms[i].split())
            term_j_words = set(normalized_terms[j].split())
            
            # Conservative lexical overlap gates to avoid cross-concept merges
            if term_i_words and term_j_words:
                word_overlap = len(term_i_words & term_j_words) / max(len(term_i_words), len(term_j_words))
                # If extremely high lexical overlap, treat as reorder/format variant
                if word_overlap >= 0.9:
                    union(i, j)
                    continue
            
            # Vector similarity check + AI adjudication when needed
            sim = _cos(vectors[i], vectors[j])
            if sim >= MERGE_THRESHOLD:
                # Light lexical gate before invoking AI: either token Jaccard >= 0.4 or shared-token ratio >= 0.2
                shared = len(term_i_words & term_j_words) if term_i_words and term_j_words else 0
                jacc = (len(term_i_words & term_j_words) / len(term_i_words | term_j_words)) if (term_i_words and term_j_words) else 0.0
                shared_ratio = (shared / min(len(term_i_words), len(term_j_words))) if (term_i_words and term_j_words and min(len(term_i_words), len(term_j_words))>0) else 0.0

                if jacc >= 0.4 or shared_ratio >= 0.2:
                    # Use AI adjudication only when NOT obvious variants
                    if not are_obvious_variants(normalized_terms[i], normalized_terms[j]):
                        def_i = constructs[i].get('definition') or normalized_terms[i]
                        def_j = constructs[j].get('definition') or normalized_terms[j]
                        is_same, conf = llm_adjudicate_constructs(def_i, def_j)
                        if is_same and conf >= 0.95:
                            union(i, j)
                    else:
                        union(i, j)

    # Build clusters
    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    # Frequency from relationships for canonical selection
    rels = data.get('relationships') or []
    freq = {}
    def _bump(term: str):
        if not term:
            return
        t = clean_construct_name(term).strip().lower()
        if t:
            freq[t] = freq.get(t, 0) + 1

    for r in rels:
        _bump(r.get('subject_term'))
        _bump(r.get('object_term'))
        for m in (r.get('moderators') or []) + (r.get('mediators') or []):
            _bump(m)
        for c in (r.get('controls') or []) or []:
            _bump(c)

    # Choose canonical per cluster
    canonical_map = {}
    cluster_canonical = {}
    for root, members in clusters.items():
        # candidates
        candidates = []
        for i in members:
            term_i = normalized_terms[i]
            definition_i = (constructs[i].get('definition') or '')
            candidates.append((term_i, len(definition_i or ''), freq.get(term_i, 0)))
        # prefer higher relationship freq, then longer definition, then shorter term
        candidates.sort(key=lambda x: (-x[2], -x[1], len(x[0])))
        canonical = candidates[0][0]
        cluster_canonical[root] = canonical
        for i in members:
            canonical_map[normalized_terms[i]] = canonical

    # Rebuild constructs: one per canonical
    merged = {}
    for c in constructs:
        t_norm = clean_construct_name(c.get('term') or '').strip().lower()
        canon = canonical_map.get(t_norm, t_norm)
        best = merged.get(canon)
        # prefer richer definition
        if not best:
            merged[canon] = {
                'term': canon,
                'definition': c.get('definition'),
                'context_snippet': c.get('context_snippet')
            }
        else:
            curr_def = best.get('definition') or ''
            new_def = c.get('definition') or ''
            if len(new_def) > len(curr_def):
                best['definition'] = c.get('definition')
                best['context_snippet'] = c.get('context_snippet')

    data['constructs'] = list(merged.values())
    
    # Log deduplication results
    original_count = len(constructs)
    final_count = len(merged)
    merged_count = original_count - final_count
    if merged_count > 0:
        print(f"  Merged {merged_count} duplicate constructs: {original_count} -> {final_count}")
        for original_term, canonical_term in canonical_map.items():
            if original_term != canonical_term:
                print(f"    '{original_term}' -> '{canonical_term}'")
    else:
        print(f"  No duplicate constructs found: {original_count} constructs")

    # Remap relationships and measurements to canonical terms
    def _remap_term(t: Optional[str]) -> Optional[str]:
        if not t:
            return t
        tn = clean_construct_name(t).strip().lower()
        return canonical_map.get(tn, tn)

    for r in rels:
        r['subject_term'] = _remap_term(r.get('subject_term'))
        r['object_term'] = _remap_term(r.get('object_term'))
        if r.get('moderators'):
            r['moderators'] = [_remap_term(m) for m in r.get('moderators') if m]
        if r.get('mediators'):
            r['mediators'] = [_remap_term(m) for m in r.get('mediators') if m]
        if r.get('controls'):
            r['controls'] = [_remap_term(m) for m in (r.get('controls') or []) if m]

    meas = data.get('measurements') or []
    for m in meas:
        m['construct_term'] = _remap_term(m.get('construct_term'))

    return data

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    logger = logging.getLogger('graph_population')
    
    logger.info(f"Reading PDF: {Path(pdf_path).name}")
    try:
        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        logger.info(f"PDF has {page_count} pages")
        
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            if i < 3:  # Log first 3 pages for debugging
                logger.debug(f"Page {i+1} text length: {len(page_text)} characters")
            elif i == 3:
                logger.debug(f"... and {page_count - 3} more pages")
        
        text = "".join(text_parts)
        cleaned_text = re.sub(r'\s+', ' ', text)
        
        logger.info(f"Text extraction completed: {len(text)} -> {len(cleaned_text)} characters (cleaned)")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {Path(pdf_path).name}: {e}")
        return None

def process_pdf_concurrent(pdf_path, vector_db, graph):
    """
    Process a single PDF file with all the extraction and graph population logic.
    This function is designed to be called concurrently.
    """
    logger = logging.getLogger('graph_population')
    
    try:
        logger.info(f"{'='*50}")
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"{'='*50}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if text:
            logger.info(f"Successfully extracted text from PDF: {len(text)} characters")
            
            # Extract knowledge using LLM
            extracted_data = get_info_from_llm(text, Path(pdf_path).name)
            if extracted_data:
                logger.info(f"LLM extraction successful for {Path(pdf_path).name}")

                # Persist artifacts for traceability under output/<hash8>/
                try:
                    # Use hash of file content as folder name for determinism
                    with open(pdf_path, 'rb') as rf:
                        h = hashlib.md5(rf.read()).hexdigest()
                    run_dir = OUTPUT_DIR / h
                    run_dir.mkdir(parents=True, exist_ok=True)

                    # Save JSON output (leave source PDF in-place; input is read-only mount)
                    json_out_path = run_dir / "extraction.json"
                    with open(json_out_path, 'w', encoding='utf-8') as jf:
                        json.dump(extracted_data, jf, ensure_ascii=False, indent=2)
                    # Do not move or modify input PDFs; processing status is tracked in Neo4j (:IngestedFile)
                except Exception as art_err:
                    logger.warning(f"Failed to persist artifacts: {art_err}")
                
                # Intra-paper deduplication
                logger.info(f"Starting intra-paper deduplication for {Path(pdf_path).name}")
                extracted_data = deduplicate_constructs_in_paper(extracted_data)
                logger.info(f"Intra-paper deduplication completed for {Path(pdf_path).name}")
                
                # Step 1: Populate graph
                logger.info(f"Starting graph population for {Path(pdf_path).name}")
                populate_graph_blueprint_compliant(graph, extracted_data, Path(pdf_path).name, vector_db)
                logger.info(f"Graph population completed for {Path(pdf_path).name}")
                
                # Step 2: Entity resolution
                if vector_db:
                    logger.info(f"Starting entity resolution for {Path(pdf_path).name}")
                    resolve_new_constructs_enhanced(graph, extracted_data, Path(pdf_path).name, vector_db)
                    logger.info(f"Entity resolution completed for {Path(pdf_path).name}")
                else:
                    logger.warning(f"Vector database not available, skipping entity resolution for {Path(pdf_path).name}")
                
                logger.info(f"Successfully processed: {Path(pdf_path).name}")
                return True
            else:
                logger.error(f"Failed to extract data from '{Path(pdf_path).name}'")
                return False
        else:
            logger.error(f"Failed to extract text from '{Path(pdf_path).name}'")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {Path(pdf_path).name}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        return False
    finally:
        logger.info(f"{'-' * 50}")

def get_info_from_llm(text, filename):
    """
    Uses OpenAI GPT-5 to extract detailed, structured information based on the centralized prompt.
    This reads the prompt from scripts/prompts/extraction.txt for consistency across all scripts.
    """
    logger = logging.getLogger('llm_extraction')
    
    logger.info(f"Starting LLM extraction for file: {filename}")
    logger.info(f"Text length: {len(text) if text else 0} characters")
    
    # Read the centralized prompt from extraction.txt
    prompt_file_path = Path(__file__).parent / "prompts" / "extraction.txt"
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        logger.info(f"Successfully loaded prompt from: {prompt_file_path}")
        logger.debug(f"Prompt length: {len(prompt_content)} characters")
    except FileNotFoundError:
        logger.warning(f"Prompt file not found at {prompt_file_path}")
        logger.info("Falling back to default prompt...")
        # Fallback to a minimal prompt if file not found
        prompt_content = """# ROLE AND GOAL
You are a research analyst extracting structured information from academic papers.
Output valid JSON only.

<PAPER_TEXT>
{text}
</PAPER_TEXT>"""
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        logger.info("Falling back to default prompt...")
        prompt_content = """# ROLE AND GOAL
You are a research analyst extracting structured information from academic papers.
Output valid JSON only.

<PAPER_TEXT>
{text}
</PAPER_TEXT>"""

    # Replace the placeholder with actual text
    prompt = prompt_content.replace("<PAPER_TEXT>", "<PAPER_TEXT>\n" + (text if text else "") + "\n")
    logger.debug(f"Final prompt length: {len(prompt)} characters")
    
    # Log the first 500 characters of the prompt for debugging
    logger.debug(f"Prompt preview: {prompt[:500]}...")
    
    try:
        logger.info(f"Calling OpenAI API with model: gpt-5-mini")
        start_time = time.time()
        
        # Try to use reasoning parameters if the model supports them
        response = _safe_chat_completion(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a research assistant skilled in knowledge extraction. Output only valid JSON based on the provided schema."},
                {"role": "user", "content": prompt}
            ]
        )
        
        api_time = time.time() - start_time
        logger.info(f"OpenAI API completed in {api_time:.2f}s")
        
        # Minimal handling; avoid logging raw content
        choice0 = (response.choices[0] if getattr(response, 'choices', None) else None)
        message0 = getattr(choice0, 'message', None) if choice0 is not None else None
        response_content = getattr(message0, 'content', None)
        # Do not log response content to keep logs compact
        
        # Parse JSON response
        try:
            if not isinstance(response_content, str) or not response_content.strip():
                raise json.JSONDecodeError("empty or non-string content", doc="", pos=0)
            parsed_data = json.loads(response_content)
            logger.info(f"Successfully parsed JSON response")
            
            # Log extracted data summary
            if isinstance(parsed_data, dict):
                constructs_count = len(parsed_data.get('constructs', []))
                relationships_count = len(parsed_data.get('relationships', []))
                measurements_count = len(parsed_data.get('measurements', []))
                theories_count = len(parsed_data.get('core_theories', []))
                
                logger.info(f"Extraction summary for {filename}: constructs={constructs_count}, relationships={relationships_count}, measurements={measurements_count}, theories={theories_count}")
                
                # Log paper metadata if available
                # Skip verbose metadata logging
                
                # Log detailed construct information
                # Skip detailed lists to keep logs small
                
                # Log detailed relationship information
                # Skip detailed lists
            
            # Do not mirror full JSON into logs
            
            return parsed_data
            
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse JSON response: {json_error}")
            logger.error(f"Response content: {response_content}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        return None

# --- 3. LLM ADJUDICATION FOR ENTITY RESOLUTION ---

def llm_adjudicate_constructs(def_a: str, def_b: str) -> tuple[bool, float]:
    """
    Use LLM to determine if two construct definitions denote the same concept and return a confidence.
    Returns (is_same_construct, confidence [0.0-1.0]).
    """
    prompt_parts = [
        "# ROLE\n",
        "You are a meticulous ontology reviewer. Decide if two academic constructs denote the SAME underlying concept.\n\n",
        "# RULES\n",
        "- Use ONLY the provided definitions/contexts; do not assume external knowledge.\n",
        "- First extract discriminative signals (scope/constraints): activity type, subject/object, boundary conditions, measurement/operationalization, domain/context, causal role.\n",
        "- If they are obvious surface-form variants (plural/singular, hyphenation, word order, prepositions, case), treat as SAME.\n",
        "- If key constraints differ (e.g., different activity types, subjects, boundary conditions, or measurement logics), treat as DIFFERENT.\n",
        "- Be conservative: when information is thin/ambiguous, prefer DIFFERENT.\n\n",
        "# INPUT A\n",
        str(def_a),
        "\n\n# INPUT B\n",
        str(def_b),
        "\n\n# OUTPUT (JSON ONLY)\n",
        '{\n  "is_same_construct": true|false,\n  "confidence": 0.00-1.00,\n  "variant_type": "surface"|"semantic"|"distinct",\n  "key_factors": ["...", "..."],\n  "rationale": "Concise justification referencing constraints"\n}\n'
    ]
    prompt = "".join(prompt_parts)

    try:
        response = _safe_chat_completion(
            model="gpt-5-nano",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an academic expert performing ontological analysis. Output only valid JSON. Temperature must be 0."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        is_same = bool(result.get("is_same_construct", False))
        confidence = float(result.get("confidence", 0.0))
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        return is_same, confidence
    except Exception as e:
        print(f"Error in LLM adjudication: {e}")
        return False, 0.0

def llm_adjudicate_theories(name_a: str, name_b: str, ctx_a: str = '', ctx_b: str = '') -> bool:
    """Use LLM to judge if two theory references denote the same theory."""
    prompt = f"""# ROLE AND GOAL
You are an expert in management and social science theories. Decide if two theory references denote the SAME theory.

# INPUT A
Name: {name_a}
Context: {ctx_a}

# INPUT B
Name: {name_b}
Context: {ctx_b}

# OUTPUT
Return ONLY JSON: {{"is_same_theory": true/false}}
"""
    try:
        response = _safe_chat_completion(
            model="gpt-5-nano",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert judging theory equivalence. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        return bool(result.get("is_same_theory", False))
    except Exception as e:
        print(f"Error in LLM theory adjudication: {e}")
        return False

# --- 4. ENHANCED NEO4J POPULATION WITH ENTITY RESOLUTION ---

def clear_database(graph):
    """Deletes all nodes and relationships in the database."""
    if not RESET_DATABASE:
        print("Database reset disabled in configuration. Skipping database clearing...")
        return
    
    print("Clearing the Neo4j database...")
    graph.delete_all()

def populate_graph_blueprint_compliant(graph, data, filename, vector_db):
    """
    Populates the Neo4j graph using the exact schema from the Technical Blueprint.
    Implements all node types and relationships as specified in the blueprint.
    """
    if not data:
        print(f"Skipping '{filename}' due to missing data.")
        return

    print(f"Populating graph for '{filename}' using blueprint-compliant schema...")
    tx = graph.begin()
    
    try:
        # Step 1: Create Paper Node with all blueprint-specified properties
        metadata = data.get('paper_metadata', {})
        paper_uid = generate_paper_uid(metadata.get('title', ''), metadata.get('publication_year'))
        
        # Format title to proper title case and authors to proper case
        title = format_title_case(metadata.get('title', ''))
        
        authors = metadata.get('authors', [])
        if authors:
            # Format each author name: first letter uppercase, rest lowercase
            # Add comma + space between authors
            formatted_authors = []
            for author in authors:
                if author and isinstance(author, str):
                    # Split by spaces and capitalize first letter of each part
                    name_parts = author.strip().split()
                    formatted_parts = [part.capitalize() for part in name_parts if part]
                    formatted_authors.append(' '.join(formatted_parts))
            authors = formatted_authors
        
        # Normalize DOI by stripping any URL scheme defensively (store compact form)
        doi_raw = (metadata.get('doi') or '')
        if isinstance(doi_raw, str):
            doi_norm = re.sub(r'https?://', '', doi_raw.strip(), flags=re.IGNORECASE)
        else:
            doi_norm = doi_raw

        # Normalize journal name to Title Case defensively
        journal_raw = metadata.get('journal')
        if isinstance(journal_raw, str):
            journal_norm = ' '.join(journal_raw.strip().split()).title()
        else:
            journal_norm = journal_raw
        
        paper_props = {
            'uuid': str(uuid.uuid4()),
            'paper_uid': paper_uid,
            'title': title,
            'doi': doi_norm,
            'authors': authors,
            'publication_year': metadata.get('publication_year'),
            'journal': journal_norm,
            'research_type': metadata.get('research_type'),
            'research_context': metadata.get('research_context'),
            'is_replication_study': metadata.get('is_replication_study', False),
            'filename': filename,
            'created_at': datetime.now().isoformat()
        }
        
        # Use paper_uid as unique identifier for papers
        paper_node = tx.run(
            """
            MERGE (p:Paper {paper_uid: $props.paper_uid})
            ON CREATE SET p = $props
            ON MATCH SET p.title = $props.title,
                         p.doi = $props.doi,
                         p.authors = $props.authors,
                         p.publication_year = $props.publication_year,
                         p.journal = $props.journal,
                         p.research_type = $props.research_type,
                         p.research_context = $props.research_context,
                         p.is_replication_study = $props.is_replication_study,
                         p.filename = $props.filename,
                         p.updated_at = $props.created_at
            RETURN p
            """, props=paper_props).evaluate()

        # Step 2: Process Authors (new in blueprint)
        for author_name in metadata.get('authors', []):
            if author_name.strip():
                author_props = {
                    'uuid': str(uuid.uuid4()),
                    'full_name': author_name.strip()
                }
                tx.run(
                    """
                    MERGE (a:Author {full_name: $props.full_name})
                    ON CREATE SET a = $props
                    WITH a
                    MATCH (p:Paper {paper_uid: $paper_uid})
                    MERGE (p)-[:AUTHORED_BY]->(a)
                    """, props=author_props, paper_uid=paper_uid)

        # Step 3: Process Constructs -> Terms, Definitions, and CanonicalConstructs
        term_to_cc_map = {}
        for construct in data.get('constructs', []):
            # Convert construct term to lowercase for consistency
            term_text = clean_construct_name(construct['term']).strip().lower()
            definition_text = construct.get('definition', '')  # Default to empty string if None
            context_snippet = construct.get('context_snippet', '')
            
            # Skip constructs without definitions
            if not definition_text or definition_text.strip() == '':
                print(f"Skipping construct '{term_text}' - no definition provided")
                continue
            
            # Create Term node (MERGE to avoid duplicates)
            term_props = {
                'uuid': str(uuid.uuid4()),
                'text': term_text
            }
            tx.run(
                """
                MERGE (t:Term {text: $props.text})
                ON CREATE SET t = $props
                """, props=term_props)

            # Create/merge Definition node (unique per term_text + paper_uid + text)
            # Ensures a single definition per construct within a paper
            def_props = {
                'uuid': str(uuid.uuid4()),
                'text': definition_text,
                'context_snippet': context_snippet,
                'term_text': term_text,
                'paper_uid': paper_uid
            }
            tx.run(
                """
                MERGE (d:Definition {text: $props.text, term_text: $props.term_text, paper_uid: $props.paper_uid})
                ON CREATE SET d.uuid = $props.uuid,
                              d.context_snippet = $props.context_snippet
                """,
                props=def_props)

            # Create CanonicalConstruct (merge by preferred_name) - initially Provisional
            cc_props = {
                'uuid': str(uuid.uuid4()),
                'preferred_name': term_text,
                'status': 'Provisional',  # Default status as per blueprint
                'description': definition_text,
                'canonical_status': None,  # Will be set during entity resolution
                'active': True
            }
            tx.run(
                """
                MERGE (cc:CanonicalConstruct {preferred_name: $props.preferred_name})
                ON CREATE SET cc = $props
                """, props=cc_props)

            # Store mapping for relationship processing
            term_to_cc_map[term_text] = term_text

            # Create relationships: Term-Definition, Term-CanonicalConstruct, Definition-Paper
            # Use MERGE to avoid duplicate relationships
            tx.run(
                """
                MATCH (t:Term {text: $term_text})
                MATCH (cc:CanonicalConstruct {preferred_name: $cc_name})
                MATCH (p:Paper {paper_uid: $paper_uid})
                // Re-find the (possibly merged) Definition node by composite keys
                MATCH (d:Definition {text: $def_text, term_text: $term_text, paper_uid: $paper_uid})
                MERGE (t)-[:HAS_DEFINITION]->(d)
                MERGE (t)-[:IS_REPRESENTATION_OF]->(cc)
                MERGE (d)-[:DEFINED_IN]->(p)
                MERGE (p)-[:USES_TERM]->(t)
                """,
                term_text=term_text,
                def_text=definition_text,
                cc_name=term_text,
                paper_uid=paper_uid)

        # Step 4: Process Construct Dimensions (new in blueprint)
        for dimension in data.get('construct_dimensions', []):
            parent = dimension.get('parent_construct', '').strip().lower()
            dim = dimension.get('dimension_construct', '').strip().lower()
            if parent and dim:
                tx.run(
                    """
                    MATCH (parent:CanonicalConstruct {preferred_name: $parent})
                    MATCH (dim:CanonicalConstruct {preferred_name: $dim})
                    MERGE (parent)-[:HAS_DIMENSION]->(dim)
                    """, parent=parent, dim=dim)

        # Step 5: Process Measurements (only for constructs that appear in relationships)
        # Build a set of constructs that actually participate in relationships
        related_constructs = set()
        for rel in data.get('relationships', []):
            st = clean_construct_name(rel.get('subject_term') or '').strip().lower()
            ot = clean_construct_name(rel.get('object_term') or '').strip().lower()
            if st:
                related_constructs.add(st)
            if ot:
                related_constructs.add(ot)
            for m in rel.get('moderators', []) or []:
                m2 = clean_construct_name(m or '').strip().lower()
                if m2:
                    related_constructs.add(m2)
            for m in rel.get('mediators', []) or []:
                m2 = clean_construct_name(m or '').strip().lower()
                if m2:
                    related_constructs.add(m2)

        for measurement in data.get('measurements', []):
            # Convert construct term to lowercase for consistency
            construct_term = (measurement.get('construct_term') or '').strip().lower()
            # Drop parenthetical abbreviations from measurement name globally
            name_raw = measurement.get('name')
            if isinstance(name_raw, str):
                name_clean = re.sub(r'\s*\([^)]*\)', '', name_raw).strip()
            else:
                name_clean = name_raw
            if not construct_term or construct_term not in related_constructs:
                continue
            meas_props = {
                'uuid': str(uuid.uuid4()),
                'name': name_clean,
                'description': measurement.get('details', ''),
                'instrument': measurement.get('instrument'),
                'scale_items': json.dumps(measurement.get('scale_items')) if isinstance(measurement.get('scale_items'), (list, dict)) else measurement.get('scale_items'),
                'scoring_procedure': measurement.get('scoring_procedure'),
                'formula': measurement.get('formula'),
                'reliability': measurement.get('reliability'),
                'validity': measurement.get('validity'),
                'context_adaptations': measurement.get('context_adaptations'),
                'construct_term': construct_term,
                'paper_uid': paper_uid
            }
            
            tx.run(
                """
                // Unique per (construct_term, paper_uid, name)
                MERGE (m:Measurement {name: $props.name, construct_term: $props.construct_term, paper_uid: $props.paper_uid})
                ON CREATE SET m = $props
                WITH m
                MATCH (cc:CanonicalConstruct {preferred_name: $construct_term})
                MATCH (p:Paper {paper_uid: $paper_uid})
                MERGE (cc)-[:USES_MEASUREMENT]->(m)
                MERGE (m)-[:MEASURED_IN]->(p)
                """,
                props=meas_props,
                construct_term=construct_term,
                paper_uid=paper_uid)

        # Step 6: Process RelationshipInstances (Core of the blueprint design)
        for rel in data.get('relationships', []):
            # Convert construct terms to lowercase for consistency
            subject_term = clean_construct_name(rel.get('subject_term') or '').strip().lower()
            object_term = clean_construct_name(rel.get('object_term') or '').strip().lower()
            
            # Verify that both constructs exist
            if subject_term not in term_to_cc_map or object_term not in term_to_cc_map:
                print(f"Warning: Could not find CanonicalConstructs for relationship: {subject_term} -> {object_term}")
                continue

            # Create RelationshipInstance node with all blueprint properties
            ri_props = {
                'uuid': str(uuid.uuid4()),
                'description': f"Relationship from {subject_term} to {object_term}",
                'context_snippet': rel.get('context_snippet', ''),
                'status': rel.get('status'),
                'evidence_type': rel.get('evidence_type'),
                'effect_direction': rel.get('effect_direction'),
                'non_linear_type': rel.get('non_linear_type'),
                'is_validated_causality': rel.get('is_validated_causality', False),
                'is_meta_analysis': rel.get('is_meta_analysis', False),
                # statistical_details removed per v2 prompt; no storage
                'qualitative_finding': rel.get('qualitative_finding'),
                'supporting_quote': rel.get('supporting_quote'),
                'boundary_conditions': rel.get('boundary_conditions'),
                'replication_outcome': rel.get('replication_outcome')
            }
            
            tx.run(
                """
                CREATE (ri:RelationshipInstance)
                SET ri = $props
                """, props=ri_props)

            # Create core relationships: Paper-ESTABLISHES->RelationshipInstance
            # RelationshipInstance-HAS_SUBJECT/HAS_OBJECT->CanonicalConstruct
            tx.run(
                """
                MATCH (p:Paper {paper_uid: $paper_uid})
                MATCH (ri:RelationshipInstance {uuid: $ri_uuid})
                MATCH (subject:CanonicalConstruct {preferred_name: $subject_term})
                MATCH (object:CanonicalConstruct {preferred_name: $object_term})
                CREATE (p)-[:ESTABLISHES]->(ri)
                CREATE (ri)-[:HAS_SUBJECT]->(subject)
                CREATE (ri)-[:HAS_OBJECT]->(object)
                """, 
                paper_uid=paper_uid,
                ri_uuid=ri_props['uuid'],
                subject_term=subject_term,
                object_term=object_term)

            # Process supporting theories with resolution & canonicalization
            for theory_name in rel.get('supporting_theories', []):
                if theory_name and isinstance(theory_name, str) and theory_name.strip():
                    try:
                        cleaned_name = clean_theory_name(theory_name)
                        context_for_theory = ri_props.get('context_snippet') or ''

                        # Default canonical id from cleaned name
                        target_canonical = canonical_id_from_name(cleaned_name)

                        # Try vector-based candidate retrieval + LLM adjudication when vector DB available
                        if vector_db:
                            emb = generate_theory_embedding(cleaned_name, context_for_theory)
                            try:
                                results = vector_db.search(
                                    collection_name=QDRANT_THEORY_COLLECTION_NAME,
                                    query_vector=emb,
                                    limit=5
                                )
                            except Exception as e:
                                print(f"Theory search error: {e}")
                                results = []

                            for cand in results:
                                if cand.score < 0.85:
                                    break
                                cand_name = (cand.payload or {}).get('name') or ''
                                cand_ctx = (cand.payload or {}).get('context') or ''
                                cand_canon = (cand.payload or {}).get('canonical_id') or canonical_id_from_name(cand_name)
                                if not cand_name:
                                    continue
                                same = llm_adjudicate_theories(cleaned_name, cand_name, context_for_theory, cand_ctx)
                                if same:
                                    target_canonical = cand_canon
                                    # Prefer the longer/more descriptive display name
                                    if len(cand_name) > len(cleaned_name):
                                        cleaned_name = cand_name
                                    break

                        # Merge canonical Theory node
                        tx.run(
                            """
                                MERGE (t:Theory {canonical_id: $cid})
                                ON CREATE SET t.uuid = $uuid, t.name = $name, t.created_at = datetime()
                                """,
                                cid=target_canonical,
                                uuid=str(uuid.uuid4()),
                                name=cleaned_name,
                        )

                        # Link RI to Theory
                        tx.run(
                            """
                            MATCH (ri:RelationshipInstance {uuid: $ri_uuid})
                            MATCH (t:Theory {canonical_id: $cid})
                            MERGE (ri)-[:APPLIES_THEORY]->(t)
                            """,
                            ri_uuid=ri_props['uuid'],
                            cid=target_canonical,
                        )

                        # Upsert into vector DB for future resolution
                        if vector_db:
                            try:
                                vector_db.upsert(
                                    collection_name=QDRANT_THEORY_COLLECTION_NAME,
                                    points=[
                                        PointStruct(
                                            id=str(uuid.uuid4()),
                                            vector=generate_theory_embedding(cleaned_name, context_for_theory),
                                            payload={
                                                'canonical_id': target_canonical,
                                                'name': cleaned_name,
                                                'context': context_for_theory,
                                            }
                                        )
                                    ]
                                )
                            except Exception as e:
                                print(f"Theory upsert error: {e}")
                    except Exception as e:
                        print(f"Theory resolution error for '{theory_name}': {e}")

            # Process moderators and mediators
            for moderator_name in rel.get('moderators', []):
                if moderator_name and moderator_name.strip():
                    # Convert moderator name to lowercase for consistency
                    moderator_name_lower = clean_construct_name(moderator_name).strip().lower()
                    tx.run(
                        """
                        MATCH (ri:RelationshipInstance {uuid: $ri_uuid})
                        MATCH (moderator:CanonicalConstruct {preferred_name: $moderator_name})
                        CREATE (ri)-[:HAS_MODERATOR]->(moderator)
                        """, ri_uuid=ri_props['uuid'], moderator_name=moderator_name_lower)

            for mediator_name in rel.get('mediators', []):
                if mediator_name and mediator_name.strip():
                    # Convert mediator name to lowercase for consistency
                    mediator_name_lower = clean_construct_name(mediator_name).strip().lower()
                    tx.run(
                        """
                        MATCH (ri:RelationshipInstance {uuid: $ri_uuid})
                        MATCH (mediator:CanonicalConstruct {preferred_name: $mediator_name})
                        CREATE (ri)-[:HAS_MEDIATOR]->(mediator)
                        """, ri_uuid=ri_props['uuid'], mediator_name=mediator_name_lower)

            # Process theoretically relevant control variables
            for control_name in rel.get('controls', []) or []:
                if control_name and control_name.strip():
                    control_name_lower = clean_construct_name(control_name).strip().lower()
                    # Ensure the control construct node exists (only if it already exists from constructs)
                    if control_name_lower in term_to_cc_map:
                        tx.run(
                            """
                            MATCH (ri:RelationshipInstance {uuid: $ri_uuid})
                            MATCH (control:CanonicalConstruct {preferred_name: $control_name})
                            MERGE (ri)-[:HAS_CONTROL]->(control)
                            """, ri_uuid=ri_props['uuid'], control_name=control_name_lower)

        # Citations intentionally omitted per updated requirements
        
        tx.commit()
        print(f"Successfully populated blueprint-compliant graph for '{filename}'.")

    except Exception as e:
        print(f"Error during transaction for '{filename}', rolling back. Error: {e}")
        tx.rollback()

# --- 5. ENTITY RESOLUTION WORKFLOW ---

def resolve_new_constructs_enhanced(graph, data, filename, vector_db):
    """
    Enhanced two-stage entity resolution workflow that preserves constructs and creates similarity relationships.
    This approach maintains all original information while establishing semantic connections.
    """
    if not vector_db:
        print("Vector database not available, skipping entity resolution.")
        return

    print(f"Running enhanced entity resolution for '{filename}'...")
    
    # Vector similarity gate to trigger LLM adjudication
    SIMILARITY_THRESHOLD = 0.80
    # Confidence threshold for deciding full merge
    MERGE_CONFIDENCE = 0.95
    
    for construct in data.get('constructs', []):
        new_term_text = clean_construct_name(construct['term']).strip().lower()
        new_definition = construct.get('definition')
        
        if not new_definition:
            continue

        # --- Stage 1: Candidate Retrieval via Vector Search ---
        new_embedding = generate_embedding(new_definition)
        
        # Search for similar constructs in vector database
        try:
            search_results = vector_db.search(
                collection_name=CONFIG.qdrant.collection,
                query_vector=new_embedding,
                limit=5
            )
        except Exception as e:
            print(f"Error searching vector database: {e}")
            search_results = []
        
        found_similar = False
        did_merge = False
        for result in search_results:
            if result.score < SIMILARITY_THRESHOLD:
                break
            
            candidate_definition = result.payload.get('definition', '')
            candidate_term = result.payload.get('term', '')
            
            if not candidate_definition or not candidate_term:
                continue
            
            # --- Stage 2: Candidate Validation via LLM Adjudication ---
            print(f"LLM adjudicating: '{new_term_text}' vs '{candidate_term}'")
            is_same, confidence = llm_adjudicate_constructs(new_definition, candidate_definition)

            # Decide canonical preferred name as the shorter of the two
            shorter_name = candidate_term if len(candidate_term) <= len(new_term_text) else new_term_text
            longer_name = new_term_text if shorter_name == candidate_term else candidate_term

            if is_same and confidence >= MERGE_CONFIDENCE:
                print(f"LLM merge decision (conf={confidence:.2f}): '{new_term_text}' <-MERGE-> '{candidate_term}'")
                # Soft merge with reversible logging
                tx = graph.begin()
                try:
                    query_merge = """
                    MATCH (keep:CanonicalConstruct {preferred_name: $keep_name})
                    MATCH (drop:CanonicalConstruct {preferred_name: $drop_name})
                    // Create operation node
                    MERGE (op:MergeOperation {id: $op_id})
                    ON CREATE SET op.created_at = datetime(), op.initiator = $initiator, op.reason = $reason,
                                      op.similarity_score = $sim, op.llm_confidence = $conf, op.status = 'applied'
                    MERGE (op)-[:KEEP]->(keep)
                    MERGE (op)-[:DROP]->(drop)

                    // Rewire RelationshipInstance subject/object/moderator/mediator/control
                    CALL {
                      WITH keep, drop, op
                      MATCH (ri:RelationshipInstance)-[r1:HAS_SUBJECT]->(drop)
                      MERGE (ri)-[:HAS_SUBJECT]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'HAS_SUBJECT', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r1
                      RETURN 0 AS _
                    }
                    CALL {
                      WITH keep, drop, op
                      MATCH (ri:RelationshipInstance)-[r2:HAS_OBJECT]->(drop)
                      MERGE (ri)-[:HAS_OBJECT]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'HAS_OBJECT', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r2
                      RETURN 0 AS _
                    }
                    CALL {
                      WITH keep, drop, op
                      MATCH (ri:RelationshipInstance)-[r3:HAS_MODERATOR]->(drop)
                      MERGE (ri)-[:HAS_MODERATOR]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'HAS_MODERATOR', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r3
                      RETURN 0 AS _
                    }
                    CALL {
                      WITH keep, drop, op
                      MATCH (ri:RelationshipInstance)-[r4:HAS_MEDIATOR]->(drop)
                      MERGE (ri)-[:HAS_MEDIATOR]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'HAS_MEDIATOR', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r4
                      RETURN 0 AS _
                    }
                    CALL {
                      WITH keep, drop, op
                      MATCH (ri:RelationshipInstance)-[r5:HAS_CONTROL]->(drop)
                      MERGE (ri)-[:HAS_CONTROL]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'HAS_CONTROL', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r5
                      RETURN 0 AS _
                    }

                    // Rewire Term representation
                    CALL {
                      WITH keep, drop, op
                      MATCH (t:Term)-[r:IS_REPRESENTATION_OF]->(drop)
                      MERGE (t)-[:IS_REPRESENTATION_OF]->(keep)
                      CREATE (ew:EdgeRewire {edge_type:'IS_REPRESENTATION_OF', term_text: t.text, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE r
                      RETURN 0 AS _
                    }

                    // Rewire measurements
                    CALL {
                      WITH keep, drop, op
                      MATCH (drop)-[um:USES_MEASUREMENT]->(m:Measurement)
                      MERGE (keep)-[:USES_MEASUREMENT]->(m)
                      CREATE (ew:EdgeRewire {edge_type:'USES_MEASUREMENT', measurement_name: m.name, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
                      MERGE (ew)-[:OF_OPERATION]->(op)
                      DELETE um
                      RETURN 0 AS _
                    }

                    // Copy over best description if needed
                    WITH keep, drop
                    CALL {
                        WITH keep, drop
                        WHERE (keep.description IS NULL OR size(toString(keep.description)) < size(toString(drop.description)))
                        SET keep.description = drop.description
                        RETURN 0 AS _
                    }

                    // Mark drop inactive and alias to keep
                    SET drop.active = true // ensure field exists before flip
                    SET drop.active = false
                    MERGE (drop)-[:ALIAS_OF {active:true, merge_operation_id: $op_id, created_at: datetime()}]->(keep)
                    SET keep.preferred_name = $keep_name,
                        keep.status = 'Verified'
                    """
                    tx.run(query_merge, keep_name=shorter_name, drop_name=longer_name,
                           op_id=str(uuid.uuid4()), initiator='auto', reason='LLM adjudicated same construct',
                           sim=result.score, conf=confidence)
                    tx.commit()
                    found_similar = True
                    did_merge = True
                    print(f"Merged constructs into canonical '{shorter_name}' (soft, reversible)")
                    break
                except Exception as e:
                    print(f"Error during merge: {e}")
                    tx.rollback()
            elif is_same:
                # Same concept but below merge confidence: mark as similar (bidirectional), keep both
                print(f"LLM similar (conf={confidence:.2f}): '{new_term_text}' ~ '{candidate_term}'")
                tx = graph.begin()
                try:
                    query_create_similarity = """
                    MATCH (a:CanonicalConstruct {preferred_name: $a_name})
                    MATCH (b:CanonicalConstruct {preferred_name: $b_name})
                    MERGE (a)-[:IS_SIMILAR_TO {similarity_score: $sim, llm_confidence: $conf, relationship_type: 'synonym', created_at: datetime()}]->(b)
                    MERGE (b)-[:IS_SIMILAR_TO {similarity_score: $sim, llm_confidence: $conf, relationship_type: 'synonym', created_at: datetime()}]->(a)
                    """
                    tx.run(query_create_similarity, a_name=new_term_text, b_name=candidate_term, sim=result.score, conf=confidence)
                    tx.commit()
                    found_similar = True
                    print(f"Created similarity links between '{new_term_text}' and '{candidate_term}'")
                    break
                except Exception as e:
                    print(f"Error creating similarity relationship: {e}")
                    tx.rollback()
        
        # Add to vector database if NOT merged (i.e., new or similar -> keep its own embedding)
        if not did_merge:
            try:
                vector_db.upsert(
                    collection_name=CONFIG.qdrant.collection,
                    points=[
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=new_embedding,
                            payload={
                                'term': new_term_text,
                                'definition': new_definition,
                                'paper': filename
                            }
                        )
                    ]
                )
                print(f"Added '{new_term_text}' to vector database")
            except Exception as e:
                print(f"Error adding to vector database: {e}")

        if not found_similar:
            # If no similar construct found, finalize as independent
            tx = graph.begin()
            try:
                query_finalize = """
                MATCH (cc:CanonicalConstruct)<-[:IS_REPRESENTATION_OF]-(t:Term {text: $term_text})
                WHERE cc.status = 'Provisional'
                SET cc.status = 'Verified',
                    cc.canonical_status = 'primary'
                """
                tx.run(query_finalize, term_text=new_term_text)
                tx.commit()
                print(f"Finalized '{new_term_text}' as independent construct")
                    
            except Exception as e:
                print(f"Error finalizing construct: {e}")
                tx.rollback()

# --- 6. MAIN EXECUTION ---
def main():
    # Initialize logging system
    loggers = setup_logging()
    main_logger = logging.getLogger()
    
    main_logger.info("--- Starting Blueprint-Compliant Knowledge Graph Build Process ---")
    main_logger.info(f"Process started at: {datetime.now().isoformat()}")
    main_logger.info(f"Database reset mode: {'ENABLED' if RESET_DATABASE else 'DISABLED (incremental processing)'}")
    
    # Setup vector database
    main_logger.info("Setting up vector database...")
    vector_db = setup_vector_database()
    if not vector_db:
        main_logger.warning("Failed to setup vector database. Continuing without entity resolution...")
    
    try:
        graph = Graph(CONFIG.neo4j.uri, auth=(CONFIG.neo4j.user, CONFIG.neo4j.password))
        graph.run("RETURN 1")
        main_logger.info("Successfully connected to Neo4j.")
    except Exception as e:
        main_logger.error(f"Failed to connect to Neo4j. Error: {e}")
        return

    # Load previously processed files
    processed_files = load_processed_files()
    main_logger.info("File tracking disabled: using move-to-output strategy")
    
    # Clear database only if configured to do so
    if RESET_DATABASE:
        main_logger.info("Clearing existing database...")
        clear_database(graph)
        # Reset processed files tracking when clearing database
        processed_files.clear()
        save_processed_files(processed_files)
        main_logger.info("Database cleared and processed files tracking reset")
    else:
        main_logger.info("Incremental processing mode: keeping existing database content")

    # Get current PDFs (anything present is considered unprocessed)
    all_pdfs, unprocessed_pdfs = get_unprocessed_files(PDF_FOLDER, set())
    
    main_logger.info(f"Total PDF files found: {len(all_pdfs)}")
    main_logger.info(f"Already processed: {len(processed_files)}")
    main_logger.info(f"New files to process: {len(unprocessed_pdfs)}")
    
    if not unprocessed_pdfs:
        main_logger.info("No new files to process. All files have been processed already.")
        return
        
    # Log all new PDF files for reference
    for i, pdf_path in enumerate(unprocessed_pdfs):
        main_logger.debug(f"New PDF {i+1}: {pdf_path}")
    
    # Process new PDFs concurrently using ThreadPoolExecutor
    main_logger.info(f"Starting concurrent processing of {len(unprocessed_pdfs)} new files with {MAX_CONCURRENT_REQUESTS} workers...")
    
    successful_count = 0
    failed_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        # Submit all new tasks
        main_logger.info("Submitting all new PDF processing tasks...")
        future_to_pdf = {
            executor.submit(process_pdf_concurrent, pdf_path, vector_db, graph): pdf_path 
            for pdf_path in unprocessed_pdfs
        }
        
        # Process completed tasks
        main_logger.info("Processing completed tasks...")
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    successful_count += 1
                    # Add to processed files set
                    processed_files.add(str(pdf_path))
                    main_logger.info(f"Task completed successfully: {Path(pdf_path).name}")
                else:
                    failed_count += 1
                    main_logger.warning(f"Task failed: {Path(pdf_path).name}")
            except Exception as e:
                main_logger.error(f"Exception occurred while processing {Path(pdf_path).name}: {e}")
                failed_count += 1
            
            # Add small delay to avoid overwhelming the API
            time.sleep(REQUEST_DELAY)
    
    # No need to persist processed files; inputs are moved out
    
    total_time = time.time() - start_time
    main_logger.info(f"\nConcurrent processing completed in {total_time:.2f} seconds!")
    main_logger.info(f"Successful: {successful_count}")
    main_logger.info(f"Failed: {failed_count}")
    main_logger.info(f"Total new files: {len(unprocessed_pdfs)}")
    main_logger.info(f"Success rate: {(successful_count/len(unprocessed_pdfs)*100):.1f}%")
    main_logger.info(f"Total processed files (cumulative): {len(processed_files)}")

    main_logger.info("\n--- Blueprint-Compliant Knowledge Graph Build Process Finished ---")
    if RESET_DATABASE:
        main_logger.info("Database was reset and completely rebuilt.")
    else:
        main_logger.info("Database was incrementally updated with new content.")
    main_logger.info("Entity resolution workflow has been executed.")
    main_logger.info("You can now run visualize_graph.py to explore the network.")
    main_logger.info(f"Process completed at: {datetime.now().isoformat()}")
    
    # Log final summary
    main_logger.info("=== FINAL SUMMARY ===")
    main_logger.info(f"Total processing time: {total_time:.2f} seconds")
    main_logger.info(f"New files processed: {len(unprocessed_pdfs)}")
    main_logger.info(f"Successful: {successful_count}")
    main_logger.info(f"Failed: {failed_count}")
    main_logger.info(f"Success rate: {(successful_count/len(unprocessed_pdfs)*100):.1f}%")
    main_logger.info(f"Total cumulative files processed: {len(processed_files)}")
    main_logger.info("=== END SUMMARY ===")

# --- 7. BACKEND OPS (update/reembed/re-resolve/merge/rollback) ---

def update_construct_description(graph: Graph, construct_name: str, new_description: str, *, editor: str = "system", reason: str | None = None, auto_merge: bool = False, similarity_threshold: float = 0.80, merge_confidence_threshold: float = 0.95, vector_db: Any | None = None) -> dict:
    """Version the construct description, re-embed, re-resolve candidates, and optionally auto-merge.

    - construct_name should be the CanonicalConstruct.preferred_name (lowercased/cleaned externally).
    - Returns a ChangeSet-like dict with revisions, embeddings, candidates, and merges.
    """
    term = clean_construct_name(construct_name).strip().lower()
    change_set: Dict[str, Any] = {"revisions": [], "embedding": None, "candidates": [], "merges": []}

    tx = graph.begin()
    try:
        # Ensure CC exists
        cc = tx.run("""
            MATCH (cc:CanonicalConstruct {preferred_name: $name})
            RETURN cc
        """, name=term).evaluate()
        if not cc:
            raise ValueError(f"CanonicalConstruct not found: {term}")

        # Create revision node and link
        rev_id = str(uuid.uuid4())
        tx.run("""
            MERGE (cc:CanonicalConstruct {preferred_name: $name})
            MERGE (rev:ConstructRevision {revision_id: $rev_id})
            ON CREATE SET rev.description = $desc, rev.created_at = datetime(), rev.editor = $editor, rev.reason = $reason
            MERGE (cc)-[:HAS_REVISION]->(rev)
            SET cc.description = $desc
        """, name=term, rev_id=rev_id, desc=new_description, editor=editor, reason=reason)
        change_set["revisions"].append({"revision_id": rev_id})

        # Re-embed
        if vector_db:
            emb = generate_embedding(new_description or "")
            try:
                vector_db.upsert(
                    collection_name=CONFIG.qdrant.collection,
                    points=[
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=emb,
                            payload={"term": term, "definition": new_description, "revision_id": rev_id}
                        )
                    ]
                )
                change_set["embedding"] = {"dimension": len(emb), "revision_id": rev_id}
            except Exception as e:
                change_set["embedding_error"] = str(e)

        # Re-resolve (search similar and adjudicate)
        candidates: List[Dict[str, Any]] = []
        if vector_db and new_description:
            try:
                search_results = vector_db.search(collection_name=CONFIG.qdrant.collection, query_vector=emb, limit=5)
            except Exception:
                search_results = []
            for r in search_results:
                if r.score < similarity_threshold:
                    break
                cand_term = (r.payload or {}).get("term")
                cand_def = (r.payload or {}).get("definition", "")
                if not cand_term or cand_term == term:
                    continue
                is_same, conf = llm_adjudicate_constructs(new_description, cand_def)
                candidates.append({"term": cand_term, "score": r.score, "is_same": is_same, "confidence": conf})
                if auto_merge and is_same and conf >= merge_confidence_threshold:
                    # prefer shorter name as keep
                    keep = cand_term if len(cand_term) <= len(term) else term
                    drop = term if keep == cand_term else cand_term
                    op_id = _apply_soft_merge(graph, keep, drop, similarity=r.score, confidence=conf, initiator=editor, reason=reason or "auto_merge via update")
                    change_set["merges"].append({"operation_id": op_id, "keep": keep, "drop": drop})
            change_set["candidates"] = candidates

        tx.commit()
    except Exception as e:
        tx.rollback()
        raise


def _apply_soft_merge(graph: Graph, keep_name: str, drop_name: str, *, similarity: float, confidence: float, initiator: str = "system", reason: str = "") -> str:
    """Internal helper to apply reversible soft-merge, mirroring main entity-resolution merge logic."""
    op_id = str(uuid.uuid4())
    tx = graph.begin()
    try:
        tx.run(
            """
            MATCH (keep:CanonicalConstruct {preferred_name: $keep_name})
            MATCH (drop:CanonicalConstruct {preferred_name: $drop_name})
            MERGE (op:MergeOperation {id: $op_id})
            ON CREATE SET op.created_at = datetime(), op.initiator = $initiator, op.reason = $reason,
                          op.similarity_score = $sim, op.llm_confidence = $conf, op.status = 'applied'
            MERGE (op)-[:KEEP]->(keep)
            MERGE (op)-[:DROP]->(drop)

            CALL {
              WITH keep, drop, op
              MATCH (ri:RelationshipInstance)-[r1:HAS_SUBJECT]->(drop)
              MERGE (ri)-[:HAS_SUBJECT]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'HAS_SUBJECT', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r1
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (ri:RelationshipInstance)-[r2:HAS_OBJECT]->(drop)
              MERGE (ri)-[:HAS_OBJECT]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'HAS_OBJECT', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r2
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (ri:RelationshipInstance)-[r3:HAS_MODERATOR]->(drop)
              MERGE (ri)-[:HAS_MODERATOR]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'HAS_MODERATOR', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r3
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (ri:RelationshipInstance)-[r4:HAS_MEDIATOR]->(drop)
              MERGE (ri)-[:HAS_MEDIATOR]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'HAS_MEDIATOR', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r4
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (ri:RelationshipInstance)-[r5:HAS_CONTROL]->(drop)
              MERGE (ri)-[:HAS_CONTROL]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'HAS_CONTROL', ri_uuid: ri.uuid, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r5
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (t:Term)-[r:IS_REPRESENTATION_OF]->(drop)
              MERGE (t)-[:IS_REPRESENTATION_OF]->(keep)
              CREATE (ew:EdgeRewire {edge_type:'IS_REPRESENTATION_OF', term_text: t.text, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH keep, drop, op
              MATCH (drop)-[um:USES_MEASUREMENT]->(m:Measurement)
              MERGE (keep)-[:USES_MEASUREMENT]->(m)
              CREATE (ew:EdgeRewire {edge_type:'USES_MEASUREMENT', measurement_name: m.name, prev_target_id: drop.uuid, new_target_id: keep.uuid, at: datetime()})
              MERGE (ew)-[:OF_OPERATION]->(op)
              DELETE um
              RETURN 0 AS _
            }

            WITH keep, drop
            CALL {
                WITH keep, drop
                WHERE (keep.description IS NULL OR size(toString(keep.description)) < size(toString(drop.description)))
                SET keep.description = drop.description
                RETURN 0 AS _
            }

            SET drop.active = coalesce(drop.active, true)
            SET drop.active = false
            MERGE (drop)-[:ALIAS_OF {active:true, merge_operation_id: $op_id, created_at: datetime()}]->(keep)
            SET keep.status = 'Verified'
            """,
            keep_name=keep_name, drop_name=drop_name, op_id=op_id,
            initiator=initiator, reason=reason, sim=similarity, conf=confidence
        )
        tx.commit()
    except Exception:
        tx.rollback()
        raise
    return op_id


def rollback_merge(graph: Graph, merge_operation_id: str) -> Dict[str, Any]:
    """Rollback a soft merge by replaying EdgeRewire logs in reverse."""
    tx = graph.begin()
    try:
        # Restore edges
        tx.run(
            """
            MATCH (op:MergeOperation {id: $op_id})
            MATCH (op)-[:DROP]->(drop:CanonicalConstruct)
            MATCH (op)-[:KEEP]->(keep:CanonicalConstruct)

            // Reverse rewires per logged edges
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_SUBJECT'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_SUBJECT]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_SUBJECT]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_OBJECT'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_OBJECT]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_OBJECT]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_MODERATOR'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_MODERATOR]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_MODERATOR]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_MEDIATOR'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_MEDIATOR]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_MEDIATOR]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_CONTROL'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_CONTROL]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_CONTROL]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'IS_REPRESENTATION_OF'})
              MATCH (t:Term)-[r:IS_REPRESENTATION_OF]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (t)-[:IS_REPRESENTATION_OF]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'USES_MEASUREMENT'})
              MATCH (:CanonicalConstruct {uuid: ew.new_target_id})-[r:USES_MEASUREMENT]->(m:Measurement {name: ew.measurement_name})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (back)-[:USES_MEASUREMENT]->(m)
              DELETE r
              RETURN 0 AS _
            }

            // Reactivate drop, disable alias
            SET drop.active = true
            MATCH (drop)-[al:ALIAS_OF]->(keep)
            SET al.active = false
            SET op.status = 'rolled_back', op.rolled_back_at = datetime()
            """,
            op_id=merge_operation_id
        )
        tx.commit()
        return {"rolled_back": True}
    except Exception:
        tx.rollback()
        raise


# --- Relationship editing ops ---
def update_relationship_instance(
    graph: Graph,
    ri_uuid: str,
    *,
    props: Optional[Dict[str, Any]] = None,
    role_changes: Optional[Dict[str, Any]] = None,
    editor: str = "system",
    reason: str | None = None,
    auto_create_construct: bool = False,
) -> Dict[str, Any]:
    """Edit a RelationshipInstance in a reversible way.

    - props: partial updates for scalar fields (status, evidence_type, effect_direction, ...)
    - role_changes: {
        subject, object,
        add_moderators: [], remove_moderators: [],
        add_mediators: [], remove_mediators: [],
        add_controls: [], remove_controls: []
      }
    """
    props = props or {}
    role_changes = role_changes or {}
    op_id = str(uuid.uuid4())
    change_set: Dict[str, Any] = {"operation_id": op_id, "rewires": [], "updated_fields": []}

    def _ensure_cc(tx, name: str):
        n = clean_construct_name(name or "").strip().lower()
        if not n:
            return None
        res = tx.run("MATCH (cc:CanonicalConstruct {preferred_name: $n}) RETURN cc.uuid as id", n=n).evaluate()
        if res:
            return n
        if auto_create_construct:
            tx.run(
                """
                MERGE (cc:CanonicalConstruct {preferred_name: $n})
                ON CREATE SET cc.uuid=$uuid, cc.status='Provisional', cc.active=true
                """,
                n=n, uuid=str(uuid.uuid4())
            )
            return n
        raise ValueError(f"Construct not found: {n}")

    tx = graph.begin()
    try:
        # Snapshot current RI properties
        current = tx.run(
            """
            MATCH (ri:RelationshipInstance {uuid: $id})
            RETURN ri as ri
            """,
            id=ri_uuid
        ).evaluate()
        if not current:
            raise ValueError("RelationshipInstance not found")

        tx.run(
            """
            MATCH (ri:RelationshipInstance {uuid: $id})
            MERGE (op:RelationshipOperation {id: $op_id})
            ON CREATE SET op.type='update', op.created_at=datetime(), op.editor=$editor, op.reason=$reason, op.status='applied'
            MERGE (op)-[:TARGET]->(ri)
            MERGE (rev:RelationshipRevision {revision_id: $rev_id})
            ON CREATE SET rev.snapshot=properties(ri), rev.created_at=datetime(), rev.editor=$editor, rev.reason=$reason
            MERGE (ri)-[:HAS_REVISION]->(rev)
            """,
            id=ri_uuid, op_id=op_id, editor=editor, reason=reason, rev_id=str(uuid.uuid4())
        )

        # Scalar props update
        if props:
            sets = []
            params = {"id": ri_uuid}
            for k, v in props.items():
                key = f"val_{k}"
                params[key] = v
                sets.append(f"ri.{k} = ${key}")
                change_set["updated_fields"].append(k)
            if sets:
                tx.run(f"MATCH (ri:RelationshipInstance {{uuid: $id}}) SET {', '.join(sets)}", **params)

        # Role changes (subject/object)
        subj = role_changes.get("subject")
        if subj is not None:
            subj = _ensure_cc(tx, subj)
            tx.run(
                """
                MATCH (ri:RelationshipInstance {uuid: $id})-[r:HAS_SUBJECT]->(old:CanonicalConstruct)
                MATCH (new:CanonicalConstruct {preferred_name: $new})
                MERGE (ri)-[:HAS_SUBJECT]->(new)
                CREATE (ew:EdgeRewire {edge_type:'HAS_SUBJECT', ri_uuid:$id, prev_target_id: old.uuid, new_target_id: new.uuid, at: datetime()})
                WITH r, ew MATCH (op:RelationshipOperation {id: $op_id}) MERGE (ew)-[:OF_OPERATION]->(op)
                DELETE r
                """,
                id=ri_uuid, new=subj, op_id=op_id
            )
            change_set["rewires"].append({"type": "HAS_SUBJECT", "new": subj})

        obj = role_changes.get("object")
        if obj is not None:
            obj = _ensure_cc(tx, obj)
            tx.run(
                """
                MATCH (ri:RelationshipInstance {uuid: $id})-[r:HAS_OBJECT]->(old:CanonicalConstruct)
                MATCH (new:CanonicalConstruct {preferred_name: $new})
                MERGE (ri)-[:HAS_OBJECT]->(new)
                CREATE (ew:EdgeRewire {edge_type:'HAS_OBJECT', ri_uuid:$id, prev_target_id: old.uuid, new_target_id: new.uuid, at: datetime()})
                WITH r, ew MATCH (op:RelationshipOperation {id: $op_id}) MERGE (ew)-[:OF_OPERATION]->(op)
                DELETE r
                """,
                id=ri_uuid, new=obj, op_id=op_id
            )
            change_set["rewires"].append({"type": "HAS_OBJECT", "new": obj})

        # Helpers for add/remove edges
        def _add_many(edge_type: str, key: str):
            for name in role_changes.get(key, []) or []:
                n = _ensure_cc(tx, name)
                tx.run(
                    f"""
                    MATCH (ri:RelationshipInstance {{uuid: $id}})
                    MATCH (cc:CanonicalConstruct {{preferred_name: $n}})
                    MERGE (ri)-[:{edge_type}]->(cc)
                    CREATE (ew:EdgeRewire {{edge_type:'{edge_type}', ri_uuid:$id, prev_target_id: null, new_target_id: cc.uuid, at: datetime()}})
                    WITH ew MATCH (op:RelationshipOperation {{id: $op_id}}) MERGE (ew)-[:OF_OPERATION]->(op)
                    """,
                    id=ri_uuid, n=n, op_id=op_id
                )
                change_set["rewires"].append({"type": edge_type, "new": n})

        def _remove_many(edge_type: str, key: str):
            for name in role_changes.get(key, []) or []:
                n = clean_construct_name(name or "").strip().lower()
                tx.run(
                    f"""
                    MATCH (ri:RelationshipInstance {{uuid: $id}})-[r:{edge_type}]->(cc:CanonicalConstruct {{preferred_name:$n}})
                    CREATE (ew:EdgeRewire {{edge_type:'{edge_type}', ri_uuid:$id, prev_target_id: cc.uuid, new_target_id: null, at: datetime()}})
                    WITH r, ew MATCH (op:RelationshipOperation {{id: $op_id}}) MERGE (ew)-[:OF_OPERATION]->(op)
                    DELETE r
                    """,
                    id=ri_uuid, n=n, op_id=op_id
                )
                change_set["rewires"].append({"type": edge_type, "remove": n})

        _add_many('HAS_MODERATOR', 'add_moderators')
        _remove_many('HAS_MODERATOR', 'remove_moderators')
        _add_many('HAS_MEDIATOR', 'add_mediators')
        _remove_many('HAS_MEDIATOR', 'remove_mediators')
        _add_many('HAS_CONTROL', 'add_controls')
        _remove_many('HAS_CONTROL', 'remove_controls')

        tx.commit()
        return change_set
    except Exception:
        tx.rollback()
        raise


def soft_delete_relationship_instance(graph: Graph, ri_uuid: str, *, editor: str = 'system', reason: str | None = None) -> Dict[str, Any]:
    op_id = str(uuid.uuid4())
    tx = graph.begin()
    try:
        tx.run(
            """
            MATCH (ri:RelationshipInstance {uuid: $id})
            SET ri.active = false
            MERGE (op:RelationshipOperation {id: $op_id})
            ON CREATE SET op.type='soft_delete', op.created_at=datetime(), op.editor=$editor, op.reason=$reason, op.status='applied'
            MERGE (op)-[:TARGET]->(ri)
            """,
            id=ri_uuid, op_id=op_id, editor=editor, reason=reason
        )
        tx.commit()
        return {"operation_id": op_id}
    except Exception:
        tx.rollback()
        raise


def restore_relationship_instance(graph: Graph, ri_uuid: str, *, editor: str = 'system', reason: str | None = None) -> Dict[str, Any]:
    op_id = str(uuid.uuid4())
    tx = graph.begin()
    try:
        tx.run(
            """
            MATCH (ri:RelationshipInstance {uuid: $id})
            SET ri.active = true
            MERGE (op:RelationshipOperation {id: $op_id})
            ON CREATE SET op.type='restore', op.created_at=datetime(), op.editor=$editor, op.reason=$reason, op.status='applied'
            MERGE (op)-[:TARGET]->(ri)
            """,
            id=ri_uuid, op_id=op_id, editor=editor, reason=reason
        )
        tx.commit()
        return {"operation_id": op_id}
    except Exception:
        tx.rollback()
        raise


def rollback_relationship_operation(graph: Graph, operation_id: str) -> Dict[str, Any]:
    """Rollback a relationship operation: restore props from snapshot and reverse EdgeRewire edges created by this op."""
    tx = graph.begin()
    try:
        # Restore props
        tx.run(
            """
            MATCH (op:RelationshipOperation {id: $op_id})-[:TARGET]->(ri:RelationshipInstance)
            MATCH (ri)-[:HAS_REVISION]->(rev:RelationshipRevision)
            WITH op, ri, rev ORDER BY rev.created_at DESC LIMIT 1
            CALL apoc.create.setProperties(ri, keys(rev.snapshot), [x IN keys(rev.snapshot) | rev.snapshot[x]]) YIELD node
            SET op.status='rolled_back', op.rolled_back_at=datetime()
            """,
            op_id=operation_id
        )

        # Reverse edges created by this op
        tx.run(
            """
            MATCH (op:RelationshipOperation {id: $op_id})
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_SUBJECT'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_SUBJECT]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_SUBJECT]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_OBJECT'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_OBJECT]->(:CanonicalConstruct {uuid: ew.new_target_id})
              MATCH (back:CanonicalConstruct {uuid: ew.prev_target_id})
              MERGE (ri)-[:HAS_OBJECT]->(back)
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_MODERATOR'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_MODERATOR]->(:CanonicalConstruct {uuid: ew.new_target_id})
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_MEDIATOR'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_MEDIATOR]->(:CanonicalConstruct {uuid: ew.new_target_id})
              DELETE r
              RETURN 0 AS _
            }
            CALL {
              WITH op
              MATCH (op)<-[:OF_OPERATION]-(ew:EdgeRewire {edge_type:'HAS_CONTROL'})
              MATCH (ri:RelationshipInstance {uuid: ew.ri_uuid})-[r:HAS_CONTROL]->(:CanonicalConstruct {uuid: ew.new_target_id})
              DELETE r
              RETURN 0 AS _
            }
            """,
            op_id=operation_id
        )

        tx.commit()
        return {"rolled_back": True}
    except Exception:
        tx.rollback()
        raise


# Keep module main entry
if __name__ == "__main__":
    main()


# --- 8. CREATE OPS: Add Construct and Relationship (user-initiated) ---

def create_construct(
    graph: Graph,
    *,
    term: str,
    definition: str,
    context_snippet: str | None = None,
    paper_uid: str | None = None,
    measurements: list | None = None,
    vector_db: Any | None = None,
    auto_merge: bool = True,
    similarity_threshold: float = 0.80,
    merge_confidence_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Create a new construct with definition (required) and optional measurements.

    - Creates Term, Definition, CanonicalConstruct, links to Paper if provided.
    - Optionally triggers entity resolution (vector search + LLM adjudication) and soft-merge.
    - Upserts the definition embedding to the constructs vector collection.
    """
    term_text = clean_construct_name(term or '').strip().lower()
    if not term_text:
        raise ValueError("term is required")
    if not definition:
        raise ValueError("definition is required")

    tx = graph.begin()
    try:
        # Ensure Paper exists if provided
        if paper_uid:
            exists = tx.run("MATCH (p:Paper {paper_uid:$pid}) RETURN p", pid=paper_uid).evaluate()
            if not exists:
                raise ValueError(f"Paper not found: {paper_uid}")

        # Create Term
        term_props = {
            'uuid': str(uuid.uuid4()),
            'text': term_text,
        }
        tx.run(
            """
            MERGE (t:Term {text: $props.text})
            ON CREATE SET t = $props
            """,
            props=term_props,
        )

        # Definition (unique per text+term_text+paper)
        def_props = {
            'uuid': str(uuid.uuid4()),
            'text': definition,
            'context_snippet': context_snippet or '',
            'term_text': term_text,
            'paper_uid': paper_uid,
        }
        tx.run(
            """
            MERGE (d:Definition {text: $props.text, term_text: $props.term_text, paper_uid: $props.paper_uid})
            ON CREATE SET d.uuid=$props.uuid, d.context_snippet=$props.context_snippet
            """,
            props=def_props,
        )

        # CanonicalConstruct (Provisional)
        cc_props = {
            'uuid': str(uuid.uuid4()),
            'preferred_name': term_text,
            'status': 'Provisional',
            'description': definition,
            'canonical_status': None,
            'active': True,
        }
        tx.run(
            """
            MERGE (cc:CanonicalConstruct {preferred_name: $props.preferred_name})
            ON CREATE SET cc = $props
            """,
            props=cc_props,
        )

        # Linkages
        link_query = """
            MATCH (t:Term {text:$term_text})
            MATCH (cc:CanonicalConstruct {preferred_name:$cc_name})
            MATCH (d:Definition {text:$def_text, term_text:$term_text, paper_uid:$paper_uid})
            MERGE (t)-[:HAS_DEFINITION]->(d)
            MERGE (t)-[:IS_REPRESENTATION_OF]->(cc)
            WITH d
            MATCH (p:Paper {paper_uid:$paper_uid})
            MERGE (d)-[:DEFINED_IN]->(p)
            MERGE (p)-[:USES_TERM]->(:Term {text:$term_text})
        """
        if paper_uid:
            tx.run(
                link_query,
                term_text=term_text,
                cc_name=term_text,
                def_text=definition,
                paper_uid=paper_uid,
            )

        # Measurements (optional)
        for m in measurements or []:
            name_raw = m.get('name')
            if not name_raw:
                continue
            name = re.sub(r'\s*\([^)]*\)', '', name_raw).strip()
            meas_props = {
                'uuid': str(uuid.uuid4()),
                'name': name,
                'description': m.get('details', ''),
                'instrument': m.get('instrument'),
                'scale_items': json.dumps(m.get('scale_items')) if isinstance(m.get('scale_items'), (list, dict)) else m.get('scale_items'),
                'scoring_procedure': m.get('scoring_procedure'),
                'formula': m.get('formula'),
                'reliability': m.get('reliability'),
                'validity': m.get('validity'),
                'context_adaptations': m.get('context_adaptations'),
                'construct_term': term_text,
                'paper_uid': paper_uid,
            }
            tx.run(
                """
                MERGE (m:Measurement {name:$props.name, construct_term:$props.construct_term, paper_uid:$props.paper_uid})
                ON CREATE SET m = $props
                WITH m
                MATCH (cc:CanonicalConstruct {preferred_name:$term})
                MERGE (cc)-[:USES_MEASUREMENT]->(m)
                """,
                props=meas_props,
                term=term_text,
            )

        tx.commit()
    except Exception as e:
        tx.rollback()
        raise

    # Entity resolution for this single construct
    result = {"term": term_text, "merged": False, "operation_id": None}
    if vector_db and definition:
        emb = generate_embedding(definition)
        try:
            search_results = vector_db.search(collection_name=CONFIG.qdrant.collection, query_vector=emb, limit=5)
        except Exception:
            search_results = []

        for r in search_results:
            if r.score < similarity_threshold:
                break
            cand_term = (r.payload or {}).get('term')
            cand_def = (r.payload or {}).get('definition', '')
            if not cand_term or cand_term == term_text:
                continue
            is_same, conf = llm_adjudicate_constructs(definition, cand_def)
            if auto_merge and is_same and conf >= merge_confidence_threshold:
                keep = cand_term if len(cand_term) <= len(term_text) else term_text
                drop = term_text if keep == cand_term else cand_term
                op_id = _apply_soft_merge(graph, keep, drop, similarity=r.score, confidence=conf, initiator='api', reason='create_construct auto-merge')
                result.update({"merged": True, "operation_id": op_id, "keep": keep, "drop": drop})
                break

        # Upsert embedding
        try:
            vector_db.upsert(
                collection_name=CONFIG.qdrant.collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={'term': term_text, 'definition': definition, 'paper': paper_uid or ''}
                    )
                ]
            )
        except Exception:
            pass

    return result


def create_relationship_instance(
    graph: Graph,
    *,
    subject: str,
    object_: str,
    status: str,
    evidence_type: str | None = None,
    effect_direction: str | None = None,
    non_linear_type: str | None = None,
    is_validated_causality: bool | None = None,
    is_meta_analysis: bool | None = None,
    theories: list | None = None,
    moderators: list | None = None,
    mediators: list | None = None,
    controls: list | None = None,
    context_snippet: str | None = None,
    description: str | None = None,
    paper_uid: str | None = None,
    auto_create_construct: bool = True,
) -> Dict[str, Any]:
    """Create a new RelationshipInstance with links and optional paper association."""
    st = clean_construct_name(subject or '').strip().lower()
    ot = clean_construct_name(object_ or '').strip().lower()
    if not st or not ot:
        raise ValueError("subject and object are required")
    if status not in ("Hypothesized", "Empirical_Result"):
        raise ValueError("status must be Hypothesized or Empirical_Result")

    tx = graph.begin()
    try:
        # Ensure constructs
        def ensure_cc(n: str):
            res = tx.run("MATCH (cc:CanonicalConstruct {preferred_name:$n}) RETURN cc", n=n).evaluate()
            if res:
                return
            if auto_create_construct:
                tx.run(
                    """
                    MERGE (cc:CanonicalConstruct {preferred_name:$n})
                    ON CREATE SET cc.uuid=$uuid, cc.status='Provisional', cc.active=true
                    """,
                    n=n, uuid=str(uuid.uuid4())
                )
            else:
                raise ValueError(f"Construct not found: {n}")

        ensure_cc(st)
        ensure_cc(ot)
        for name in (moderators or []):
            if name: ensure_cc(clean_construct_name(name).strip().lower())
        for name in (mediators or []):
            if name: ensure_cc(clean_construct_name(name).strip().lower())
        for name in (controls or []):
            if name: ensure_cc(clean_construct_name(name).strip().lower())

        # Paper check
        if paper_uid:
            ok = tx.run("MATCH (p:Paper {paper_uid:$pid}) RETURN p", pid=paper_uid).evaluate()
            if not ok:
                raise ValueError(f"Paper not found: {paper_uid}")

        ri_props = {
            'uuid': str(uuid.uuid4()),
            'description': description or f"Relationship from {st} to {ot}",
            'context_snippet': context_snippet or '',
            'status': status,
            'evidence_type': evidence_type,
            'effect_direction': effect_direction,
            'non_linear_type': non_linear_type,
            'is_validated_causality': bool(is_validated_causality) if is_validated_causality is not None else None,
            'is_meta_analysis': bool(is_meta_analysis) if is_meta_analysis is not None else None,
        }
        tx.run("CREATE (ri:RelationshipInstance) SET ri = $props", props=ri_props)

        tx.run(
            """
            MATCH (ri:RelationshipInstance {uuid:$id})
            MATCH (s:CanonicalConstruct {preferred_name:$st})
            MATCH (o:CanonicalConstruct {preferred_name:$ot})
            MERGE (ri)-[:HAS_SUBJECT]->(s)
            MERGE (ri)-[:HAS_OBJECT]->(o)
            """,
            id=ri_props['uuid'], st=st, ot=ot,
        )

        if paper_uid:
            tx.run(
                """
                MATCH (p:Paper {paper_uid:$pid})
                MATCH (ri:RelationshipInstance {uuid:$id})
                MERGE (p)-[:ESTABLISHES]->(ri)
                """,
                pid=paper_uid, id=ri_props['uuid']
            )

        for name in (moderators or []):
            m = clean_construct_name(name or '').strip().lower()
            if not m: continue
            tx.run(
                """
                MATCH (ri:RelationshipInstance {uuid:$id})
                MATCH (m:CanonicalConstruct {preferred_name:$m})
                MERGE (ri)-[:HAS_MODERATOR]->(m)
                """,
                id=ri_props['uuid'], m=m,
            )
        for name in (mediators or []):
            m = clean_construct_name(name or '').strip().lower()
            if not m: continue
            tx.run(
                """
                MATCH (ri:RelationshipInstance {uuid:$id})
                MATCH (m:CanonicalConstruct {preferred_name:$m})
                MERGE (ri)-[:HAS_MEDIATOR]->(m)
                """,
                id=ri_props['uuid'], m=m,
            )
        for name in (controls or []):
            m = clean_construct_name(name or '').strip().lower()
            if not m: continue
            tx.run(
                """
                MATCH (ri:RelationshipInstance {uuid:$id})
                MATCH (m:CanonicalConstruct {preferred_name:$m})
                MERGE (ri)-[:HAS_CONTROL]->(m)
                """,
                id=ri_props['uuid'], m=m,
            )

        # Theories (optional) as canonicalized names
        for tname in (theories or []):
            if not tname or not str(tname).strip():
                continue
            cleaned_name = clean_theory_name(tname)
            cid = canonical_id_from_name(cleaned_name)
            tx.run(
                """
                MERGE (t:Theory {canonical_id:$cid})
                ON CREATE SET t.uuid=$uuid, t.name=$name, t.created_at=datetime()
                WITH t
                MATCH (ri:RelationshipInstance {uuid:$id})
                MERGE (ri)-[:APPLIES_THEORY]->(t)
                """,
                cid=cid, uuid=str(uuid.uuid4()), name=cleaned_name, id=ri_props['uuid']
            )

        tx.commit()
        return {"ri_uuid": ri_props['uuid']}
    except Exception:
        tx.rollback()
        raise
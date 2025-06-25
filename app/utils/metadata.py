
from transformers import pipeline
from keybert import KeyBERT
from bertopic import BERTopic
import spacy
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
import numpy as np
import logging
from functools import lru_cache
import warnings
from pathlib import Path
import hashlib
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Enhanced metadata extraction class with improved error handling and caching."""
    
    def __init__(self):
        """Initialize models with lazy loading and error handling."""
        self._models = {}
        self._model_cache = {}
        
    @lru_cache(maxsize=1)
    def _get_summarizer(self):
        """Lazy load summarization model with error handling."""
        try:
            if 'summarizer' not in self._models:
                logger.info("Loading summarization model...")
                self._models['summarizer'] = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=-1  # Use CPU by default
                )
            return self._models['summarizer']
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}")
            return None
    
    @lru_cache(maxsize=1)
    def _get_keybert_model(self):
        """Lazy load KeyBERT model."""
        try:
            if 'keybert' not in self._models:
                logger.info("Loading KeyBERT model...")
                self._models['keybert'] = KeyBERT()
            return self._models['keybert']
        except Exception as e:
            logger.error(f"Failed to load KeyBERT: {e}")
            return None
    
    @lru_cache(maxsize=1)
    def _get_spacy_model(self):
        """Lazy load spaCy model."""
        try:
            if 'spacy' not in self._models:
                logger.info("Loading spaCy model...")
                self._models['spacy'] = spacy.load("en_core_web_sm")
            return self._models['spacy']
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            try:
                # Fallback to blank model
                logger.info("Trying blank spaCy model...")
                self._models['spacy'] = spacy.blank("en")
                return self._models['spacy']
            except Exception as e2:
                logger.error(f"Failed to load blank spaCy model: {e2}")
                return None
    
    @lru_cache(maxsize=1)
    def _get_topic_model(self):
        """Lazy load BERTopic model."""
        try:
            if 'topic' not in self._models:
                logger.info("Loading BERTopic model...")
                self._models['topic'] = BERTopic(verbose=False)
            return self._models['topic']
        except Exception as e:
            logger.error(f"Failed to load BERTopic: {e}")
            return None
    
    @lru_cache(maxsize=1)
    def _get_sentiment_analyzer(self):
        """Lazy load sentiment analysis model."""
        try:
            if 'sentiment' not in self._models:
                logger.info("Loading sentiment analysis model...")
                self._models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    device=-1,
                    return_all_scores=True
                )
            return self._models['sentiment']
        except Exception as e:
            logger.error(f"Failed to load sentiment analyzer: {e}")
            return None
    
    @lru_cache(maxsize=1)
    def _get_classifier(self):
        """Lazy load zero-shot classifier."""
        try:
            if 'classifier' not in self._models:
                logger.info("Loading zero-shot classifier...")
                self._models['classifier'] = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli",
                    device=-1
                )
            return self._models['classifier']
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return None

# Global instance
extractor = MetadataExtractor()

# Enhanced candidate categories with more specific types
ENTITY_LABELS = [
    "person", "organization", "technology", "project", "metric",
    "date", "location", "contact", "document", "financial", 
    "legal", "medical", "educational", "product", "service", "misc"
]

def validate_text_input(text: str) -> bool:
    """Validate input text."""
    if not text or not isinstance(text, str):
        return False
    if len(text.strip()) < 10:  # Minimum meaningful text length
        return False
    return True

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()

def classify_entity(entity: str, threshold: float = 0.3) -> str:
    """Classify entity with improved error handling and lower threshold."""
    classifier = extractor._get_classifier()
    if not classifier:
        return "misc"
    
    try:
        entity_clean = clean_text(entity)
        if len(entity_clean) < 2:
            return "misc"
            
        result = classifier(entity_clean, candidate_labels=ENTITY_LABELS, multi_label=True)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return top_label if top_score >= threshold else "misc"
    except Exception as e:
        logger.warning(f"Error classifying entity '{entity}': {e}")
        return "misc"

def group_entities_model_based(entities: List[str]) -> Dict[str, List[str]]:
    """Group entities by category with improved processing."""
    grouped = {label: [] for label in ENTITY_LABELS}
    
    if not entities:
        return grouped
    
    # Remove duplicates while preserving order
    unique_entities = list(dict.fromkeys(entities))
    
    for ent in unique_entities:
        try:
            ent_clean = clean_text(ent)
            
            # Skip very short or numeric-only entities
            if len(ent_clean) < 2 or ent_clean.isnumeric():
                continue
                
            # Skip common stop words and articles
            if ent_clean.lower() in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}:
                continue
            
            label = classify_entity(ent_clean)
            if ent_clean not in grouped[label]:  # Avoid duplicates
                grouped[label].append(ent_clean)
                
        except Exception as e:
            logger.warning(f"Error processing entity '{ent}': {e}")
            continue
    
    # Remove empty categories
    return {k: v for k, v in grouped.items() if v}

def generate_summary(text: str, summary_type: str = "comprehensive", max_chunk_size: int = 1024) -> Dict[str, str]:
    """Generate detailed, comprehensive summaries with multiple approaches."""
    summarizer = extractor._get_summarizer()
    if not summarizer or not validate_text_input(text):
        return {"detailed_summary": "Summary not available", "brief_summary": "Summary not available"}
    
    try:
        text_clean = clean_text(text)
        word_count = len(text_clean.split())
        
        # Calculate appropriate summary lengths based on text length
        if word_count < 200:
            detailed_max_length = min(100, word_count // 2)
            brief_max_length = min(50, word_count // 3)
        elif word_count < 1000:
            detailed_max_length = min(250, word_count // 3)
            brief_max_length = min(100, word_count // 6)
        elif word_count < 5000:
            detailed_max_length = min(400, word_count // 4)
            brief_max_length = min(150, word_count // 8)
        else:
            detailed_max_length = min(600, word_count // 5)
            brief_max_length = min(200, word_count // 10)
        
        detailed_min_length = max(50, detailed_max_length // 4)
        brief_min_length = max(20, brief_max_length // 4)
        
        # For shorter texts, generate summary directly
        if len(text_clean) <= max_chunk_size and word_count < 500:
            try:
                # Generate detailed summary
                detailed_result = summarizer(
                    text_clean,
                    max_length=detailed_max_length,
                    min_length=detailed_min_length,
                    do_sample=False,
                    length_penalty=1.0,
                    num_beams=4
                )
                
                # Generate brief summary
                brief_result = summarizer(
                    text_clean,
                    max_length=brief_max_length,
                    min_length=brief_min_length,
                    do_sample=False,
                    length_penalty=1.2,
                    num_beams=4
                )
                
                return {
                    "detailed_summary": detailed_result[0]["summary_text"].strip(),
                    "brief_summary": brief_result[0]["summary_text"].strip()
                }
            except Exception as e:
                logger.warning(f"Error in direct summarization: {e}")
        
        # For longer texts, use intelligent chunking
        nlp = extractor._get_spacy_model()
        chunks = []
        
        if nlp:
            try:
                # Process text in manageable sections to avoid memory issues
                text_sections = [text_clean[i:i+50000] for i in range(0, len(text_clean), 50000)]
                all_sentences = []
                
                for section in text_sections:
                    doc = nlp(section)
                    all_sentences.extend([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10])
                
                # Create meaningful chunks based on sentences
                current_chunk = ""
                for sentence in all_sentences:
                    # Check if adding this sentence would exceed chunk size
                    if len(current_chunk + sentence) <= max_chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    
            except Exception as e:
                logger.warning(f"Error in sentence-based chunking: {e}")
                # Fallback to paragraph-based chunking
                paragraphs = [p.strip() for p in text_clean.split('\n\n') if p.strip()]
                current_chunk = ""
                
                for paragraph in paragraphs:
                    if len(current_chunk + paragraph) <= max_chunk_size:
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph + "\n\n"
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        
        # Fallback to simple chunking if sentence processing fails
        if not chunks:
            chunks = [text_clean[i:i+max_chunk_size] for i in range(0, len(text_clean), max_chunk_size)]
        
        # Generate summaries for each chunk with varying detail levels
        detailed_summaries = []
        brief_summaries = []
        
        # Limit chunks to prevent memory/time issues
        max_chunks = min(15, len(chunks))
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            if len(chunk.strip()) < 100:  # Skip very short chunks
                continue
            
            chunk_word_count = len(chunk.split())
            
            # Calculate chunk-specific summary lengths
            chunk_detailed_max = min(120, max(40, chunk_word_count // 3))
            chunk_detailed_min = min(30, chunk_detailed_max // 3)
            chunk_brief_max = min(60, max(20, chunk_word_count // 5))
            chunk_brief_min = min(15, chunk_brief_max // 3)
            
            try:
                # Generate detailed chunk summary
                detailed_chunk_summary = summarizer(
                    chunk,
                    max_length=chunk_detailed_max,
                    min_length=chunk_detailed_min,
                    do_sample=False,
                    length_penalty=0.8,  # Encourage longer summaries
                    num_beams=4
                )
                detailed_summaries.append(detailed_chunk_summary[0]["summary_text"])
                
                # Generate brief chunk summary
                brief_chunk_summary = summarizer(
                    chunk,
                    max_length=chunk_brief_max,
                    min_length=chunk_brief_min,
                    do_sample=False,
                    length_penalty=1.2,  # Encourage brevity
                    num_beams=4
                )
                brief_summaries.append(brief_chunk_summary[0]["summary_text"])
                
            except Exception as e:
                logger.warning(f"Error summarizing chunk {i+1}: {e}")
                continue
        
        if not detailed_summaries:
            return {"detailed_summary": "Summary generation failed", "brief_summary": "Summary generation failed"}
        
        # Combine chunk summaries
        combined_detailed = " ".join(detailed_summaries).strip()
        combined_brief = " ".join(brief_summaries).strip()
        
        # If combined summaries are too long, create final consolidated summaries
        final_detailed_summary = combined_detailed
        final_brief_summary = combined_brief
        
        # Create a comprehensive final summary if we have multiple chunk summaries
        if len(detailed_summaries) > 1 and len(combined_detailed.split()) > detailed_max_length:
            try:
                final_detailed_result = summarizer(
                    combined_detailed,
                    max_length=detailed_max_length,
                    min_length=detailed_min_length,
                    do_sample=False,
                    length_penalty=0.8,
                    num_beams=4
                )
                final_detailed_summary = final_detailed_result[0]["summary_text"].strip()
            except Exception as e:
                logger.warning(f"Error in final detailed summarization: {e}")
                # Keep the combined summary but truncate if too long
                if len(combined_detailed) > 3000:
                    final_detailed_summary = combined_detailed[:3000] + "..."
        
        if len(brief_summaries) > 1 and len(combined_brief.split()) > brief_max_length:
            try:
                final_brief_result = summarizer(
                    combined_brief,
                    max_length=brief_max_length,
                    min_length=brief_min_length,
                    do_sample=False,
                    length_penalty=1.2,
                    num_beams=4
                )
                final_brief_summary = final_brief_result[0]["summary_text"].strip()
            except Exception as e:
                logger.warning(f"Error in final brief summarization: {e}")
                # Keep the combined summary but truncate if too long
                if len(combined_brief) > 1000:
                    final_brief_summary = combined_brief[:1000] + "..."
        
        # Ensure we have meaningful summaries
        if not final_detailed_summary or len(final_detailed_summary.split()) < 10:
            final_detailed_summary = combined_detailed if combined_detailed else "Unable to generate detailed summary"
        
        if not final_brief_summary or len(final_brief_summary.split()) < 5:
            final_brief_summary = combined_brief if combined_brief else "Unable to generate brief summary"
        
        return {
            "detailed_summary": final_detailed_summary,
            "brief_summary": final_brief_summary,
            "chunk_count": len(chunks),
            "summary_statistics": {
                "detailed_word_count": len(final_detailed_summary.split()),
                "brief_word_count": len(final_brief_summary.split()),
                "compression_ratio_detailed": round(word_count / len(final_detailed_summary.split()), 2) if final_detailed_summary else 0,
                "compression_ratio_brief": round(word_count / len(final_brief_summary.split()), 2) if final_brief_summary else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive summary: {e}")
        return {"detailed_summary": "Summary generation failed", "brief_summary": "Summary generation failed"}

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment with improved accuracy and error handling."""
    sentiment_analyzer = extractor._get_sentiment_analyzer()
    if not sentiment_analyzer or not validate_text_input(text):
        return {"label": "neutral", "confidence": 0.0, "scores": {}}
    
    try:
        text_clean = clean_text(text)
        
        # Chunk text for analysis (sentiment models typically handle shorter texts better)
        chunk_size = 512
        chunks = [text_clean[i:i+chunk_size] for i in range(0, min(len(text_clean), 5120), chunk_size)]
        
        all_scores = []
        
        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
                
            try:
                result = sentiment_analyzer(chunk)
                if isinstance(result, list) and len(result) > 0:
                    chunk_scores = result[0] if isinstance(result[0], list) else result
                    all_scores.extend(chunk_scores)
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for chunk: {e}")
                continue
        
        if not all_scores:
            return {"label": "neutral", "confidence": 0.0, "scores": {}}
        
        # Aggregate scores
        score_dict = {}
        for score_info in all_scores:
            label = score_info['label'].upper()
            if label not in score_dict:
                score_dict[label] = []
            score_dict[label].append(score_info['score'])
        
        # Calculate average scores
        avg_scores = {}
        for label, scores in score_dict.items():
            avg_scores[label.lower()] = np.mean(scores)
        
        # Determine final sentiment
        if not avg_scores:
            return {"label": "neutral", "confidence": 0.0, "scores": {}}
        
        final_label = max(avg_scores, key=avg_scores.get)
        final_confidence = avg_scores[final_label]
        
        return {
            "label": final_label,
            "confidence": round(float(final_confidence), 4),
            "scores": {k: round(float(v), 4) for k, v in avg_scores.items()}
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"label": "neutral", "confidence": 0.0, "scores": {}}

def extract_advanced_structure(text: str) -> Dict[str, Any]:
    """Extract detailed document structure information."""
    nlp = extractor._get_spacy_model()
    
    if not validate_text_input(text):
        return {"error": "Invalid text input"}
    
    text_clean = clean_text(text)
    lines = text_clean.splitlines()
    paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
    
    structure = {
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "empty_lines": len([line for line in lines if not line.strip()]),
        "average_paragraph_length": np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0,
    }
    
    if nlp:
        try:
            # Process in smaller chunks to avoid memory issues
            doc = nlp(text_clean[:50000])  # Limit to first 50k characters
            sentences = list(doc.sents)
            tokens = [token for token in doc if not token.is_space]
            words = [token for token in tokens if token.is_alpha]
            
            structure.update({
                "sentence_count": len(sentences),
                "word_count": len(words),
                "token_count": len(tokens),
                "average_words_per_sentence": round(len(words) / len(sentences), 2) if sentences else 0,
                "average_sentences_per_paragraph": round(len(sentences) / len(paragraphs), 2) if paragraphs else 0,
                "lexical_diversity": round(len(set(token.text.lower() for token in words)) / len(words), 4) if words else 0,
            })
        except Exception as e:
            logger.warning(f"Error in spaCy processing: {e}")
            words = text_clean.split()
            structure.update({
                "sentence_count": text_clean.count('.') + text_clean.count('!') + text_clean.count('?'),
                "word_count": len(words),
                "token_count": len(words),
                "average_words_per_sentence": 0,
                "average_sentences_per_paragraph": 0,
                "lexical_diversity": 0,
            })
    else:
        words = text_clean.split()
        structure.update({
            "sentence_count": text_clean.count('.') + text_clean.count('!') + text_clean.count('?'),
            "word_count": len(words),
            "token_count": len(words),
            "average_words_per_sentence": 0,
            "average_sentences_per_paragraph": 0,
            "lexical_diversity": 0,
        })
    
    # Detect potential headers and formatting
    potential_headers = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and (
            line_clean.isupper() or 
            line_clean.endswith(':') or
            (len(line_clean.split()) <= 8 and not line_clean.endswith('.'))
        ):
            potential_headers.append(line_clean)
    
    structure["potential_headers"] = potential_headers[:20]  # Limit to first 20
    structure["has_bullet_points"] = bool(re.search(r'^\s*[•\-\*]\s', text_clean, re.MULTILINE))
    structure["has_numbered_lists"] = bool(re.search(r'^\s*\d+\.\s', text_clean, re.MULTILINE))
    
    return structure

def extract_keywords_enhanced(text: str, top_n: int = 25) -> Dict[str, List]:
    """Extract keywords using multiple methods."""
    keybert_model = extractor._get_keybert_model()
    nlp = extractor._get_spacy_model()
    
    result = {
        "keybert_keywords": [],
        "named_entities": [],
        "combined_keywords": []
    }
    
    if not validate_text_input(text):
        return result
    
    text_clean = clean_text(text)
    
    # KeyBERT extraction
    if keybert_model:
        try:
            keywords = keybert_model.extract_keywords(
                text_clean, 
                top_n=top_n,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                diversity=0.5
            )
            result["keybert_keywords"] = [{"keyword": k[0], "score": round(k[1], 4)} for k in keywords if len(k[0]) > 2]
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
    
    # Named entity extraction
    if nlp:
        try:
            doc = nlp(text_clean[:50000])  # Limit text length
            entities = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 2]
            result["named_entities"] = list(set(entities))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {e}")
    
    # Combine all keywords
    all_keywords = set()
    all_keywords.update([kw["keyword"] for kw in result["keybert_keywords"]])
    all_keywords.update([ent[0] for ent in result["named_entities"]])
    
    result["combined_keywords"] = sorted(list(all_keywords))
    
    return result

def calculate_content_hash(text: str) -> str:
    """Calculate content hash for duplicate detection."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def detect_language(text: str) -> str:
    """Simple language detection (can be enhanced with proper language detection library)."""
    # This is a simple implementation - consider using langdetect library for better accuracy
    english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = set(word.lower() for word in text.split()[:100])  # Check first 100 words
    english_matches = len(words & english_words)
    
    return "en" if english_matches > 3 else "unknown"

def generate_metadata(text: str, filename: str = "uploaded.pdf", file_path: str = "uploaded.pdf") -> Dict[str, Any]:
    """Generate comprehensive metadata with enhanced error handling and features."""
    
    if not validate_text_input(text):
        logger.error("Invalid text input for metadata generation")
        return {"error": "Invalid text input", "timestamp": datetime.now().isoformat()}
    
    start_time = datetime.now()
    text_clean = clean_text(text)
    
    # Basic file information
    metadata = {
        "generation_timestamp": start_time.isoformat(),
        "system_version": "3.0.0",
        "advanced_features_enabled": True,
        "filename": filename,
        "file_path": file_path,
        "file_type": Path(filename).suffix.lower() if filename else "",
        "content_hash": calculate_content_hash(text_clean),
        "processing_time_seconds": 0,  # Will be updated at the end
    }
    
    # File stats (with error handling for non-existent files)
    try:
        if os.path.exists(file_path):
            stats = os.stat(file_path)
            metadata.update({
                "file_size": stats.st_size,
                "file_size_mb": round(stats.st_size / 1e6, 3),
                "creation_date": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modification_date": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "access_date": datetime.fromtimestamp(stats.st_atime).isoformat(),
            })
        else:
            metadata.update({
                "file_size": len(text_clean.encode('utf-8')),
                "file_size_mb": round(len(text_clean.encode('utf-8')) / 1e6, 3),
                "creation_date": start_time.isoformat(),
                "modification_date": start_time.isoformat(),
                "access_date": start_time.isoformat(),
            })
    except Exception as e:
        logger.warning(f"Error getting file stats: {e}")
        metadata.update({
            "file_size": len(text_clean.encode('utf-8')),
            "file_size_mb": round(len(text_clean.encode('utf-8')) / 1e6, 3),
            "creation_date": start_time.isoformat(),
            "modification_date": start_time.isoformat(),
            "access_date": start_time.isoformat(),
        })
    
    # Basic text metrics
    metadata.update({
        "page_count": max(text_clean.count("\f") + 1, 1),
        "character_count": len(text_clean),
        "character_count_no_spaces": len(text_clean.replace(" ", "")),
        "word_count": len(text_clean.split()),
        "language": detect_language(text_clean),
    })
    
    # Document structure analysis
    try:
        metadata["document_structure"] = extract_advanced_structure(text_clean)
    except Exception as e:
        logger.error(f"Error in structure analysis: {e}")
        metadata["document_structure"] = {"error": str(e)}
    
    # Keyword and entity extraction
    try:
        keywords_data = extract_keywords_enhanced(text_clean)
        metadata["keywords"] = keywords_data
        
        # Extract entities for grouping
        all_entities = [ent[0] for ent in keywords_data["named_entities"]]
        metadata["entities"] = group_entities_model_based(all_entities)
        
    except Exception as e:
        logger.error(f"Error in keyword/entity extraction: {e}")
        metadata["keywords"] = {"error": str(e)}
        metadata["entities"] = {"error": str(e)}
    
    # Sentiment analysis
    try:
        metadata["sentiment"] = analyze_sentiment(text_clean)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        metadata["sentiment"] = {"error": str(e)}
    
    # Summary generation (Enhanced with detailed and brief versions)
    try:
        summary_result = generate_summary(text_clean)
        metadata["summary"] = summary_result
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        metadata["summary"] = {
            "detailed_summary": "Summary generation failed",
            "brief_summary": "Summary generation failed"
        }
    
    # Topic modeling
    try:
        topic_model = extractor._get_topic_model()
        if topic_model and len(text_clean.split()) > 50:
            topics, _ = topic_model.fit_transform([text_clean])
            if topics[0] != -1:  # -1 indicates no topic found
                top_topic = topic_model.get_topic(topics[0])
                metadata["topics"] = {
                    "topic_id": int(topics[0]),
                    "keywords": top_topic[:10] if top_topic else [],
                    "confidence": "Available in topic modeling results"
                }
            else:
                metadata["topics"] = "No clear topics identified"
        else:
            metadata["topics"] = "Insufficient text for topic modeling"
    except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
        metadata["topics"] = {"error": str(e)}
    
    # Calculate processing time
    end_time = datetime.now()
    metadata["processing_time_seconds"] = round((end_time - start_time).total_seconds(), 3)
    metadata["completion_timestamp"] = end_time.isoformat()
    
    logger.info(f"Metadata generation completed in {metadata['processing_time_seconds']} seconds")
    
    return metadata

# Utility functions for Streamlit integration
def format_metadata_for_display(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata for better display in Streamlit."""
    display_data = metadata.copy()
    
    # Format large numbers
    if "character_count" in display_data:
        display_data["character_count_formatted"] = f"{display_data['character_count']:,}"
    if "word_count" in display_data:
        display_data["word_count_formatted"] = f"{display_data['word_count']:,}"
    
    # Format timestamps
    for key in ["generation_timestamp", "completion_timestamp", "creation_date", "modification_date"]:
        if key in display_data:
            try:
                dt = datetime.fromisoformat(display_data[key].replace('Z', '+00:00'))
                display_data[f"{key}_formatted"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
    
    return display_data

def get_model_status() -> Dict[str, bool]:
    """Get status of all models for debugging."""
    return {
        "summarizer": extractor._get_summarizer() is not None,
        "keybert": extractor._get_keybert_model() is not None,
        "spacy": extractor._get_spacy_model() is not None,
        "topic_model": extractor._get_topic_model() is not None,
        "sentiment_analyzer": extractor._get_sentiment_analyzer() is not None,
        "classifier": extractor._get_classifier() is not None,
    }


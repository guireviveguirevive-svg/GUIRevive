"""
CodeMatchHandler - GUI Element Matching Component
Based on COSER paper (ICSE 2024)

This component implements GUI element matching between base and updated versions
using both external semantics (UI properties) and internal semantics (code snippets).
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from xml.etree import ElementTree as ET
import numpy as np


class CodeMatchHandler:
    """
    Handles matching of GUI elements between base and updated app versions.

    Uses two types of semantic information:
    1. External Semantics: text, content-desc, resource-id properties
    2. Internal Semantics: code snippets associated with GUI elements
    """

    def __init__(self,
                 external_threshold: float = 0.7,
                 use_sentence_bert: bool = True,
                 use_unixcoder: bool = True):
        """
        Initialize CodeMatchHandler.

        Args:
            external_threshold: Threshold for external semantic matching (default: 0.7)
            use_sentence_bert: Whether to use Sentence-BERT for external matching
            use_unixcoder: Whether to use UniXcoder for internal matching
        """
        self.external_threshold = external_threshold
        self.use_sentence_bert = use_sentence_bert
        self.use_unixcoder = use_unixcoder

        # Cache for semantic matching results
        self.match_cache = {}

        # Models (to be loaded)
        self.sentence_bert_model = None
        self.unixcoder_model = None

    def initialize_models(self):
        """Initialize the deep learning models for semantic matching."""
        if self.use_sentence_bert:
            self._load_sentence_bert()

        if self.use_unixcoder:
            self._load_unixcoder()

    def _load_sentence_bert(self):
        """Load Sentence-BERT model for external semantic matching."""
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading Sentence-BERT model...")
            self.sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("Sentence-BERT model loaded successfully.")
        except ImportError:
            print("Warning: sentence-transformers not installed. Install with:")
            print("pip install sentence-transformers")
            self.use_sentence_bert = False

    def _load_unixcoder(self):
        """Load UniXcoder model for internal semantic matching."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            # Load tokenizer from original UniXcoder
            print("Loading UniXcoder model...")
            self.unixcoder_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
            self.unixcoder_model = AutoModel.from_pretrained("microsoft/unixcoder-base")

            # Load fine-tuned weights if available
            local_model_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.bin")
            if os.path.exists(local_model_bin):
                print(f"Loading fine-tuned weights from {local_model_bin}...")
                state_dict = torch.load(local_model_bin, map_location="cpu")
                self.unixcoder_model.load_state_dict(state_dict, strict=False)
                print("Fine-tuned weights loaded successfully.")

            self.unixcoder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.unixcoder_model.to(self.unixcoder_device)
            self.unixcoder_model.eval()

            print(f"UniXcoder model loaded successfully on {self.unixcoder_device}.")

        except ImportError:
            print("Warning: transformers or torch not installed. Install with:")
            print("pip install transformers torch")
            self.use_unixcoder = False
        except Exception as e:
            print(f"Warning: Could not load UniXcoder: {e}")
            print("Using fallback code similarity approach.")
            self.use_unixcoder = False

    def get_external_semantic_match(self,
                                   old_element: Dict,
                                   current_screen_elements: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Find matching element based on external semantics.

        According to the paper (Section 3.4.1):
        - Uses Sentence-BERT to calculate similarity of external semantics
        - Returns element with highest similarity score
        - Only accepts match if score > threshold

        Args:
            old_element: GUI element from base version
            current_screen_elements: List of GUI elements on current screen of updated version

        Returns:
            Tuple of (matched_element, similarity_score)
        """
        if not current_screen_elements:
            return None, 0.0

        old_semantic = self._get_external_semantic(old_element)
        if not old_semantic:
            return None, 0.0

        # Check cache
        cache_key = (old_semantic, tuple(self._get_external_semantic(e) for e in current_screen_elements))
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        best_match = None
        best_score = 0.0

        if self.use_sentence_bert and self.sentence_bert_model:
            # Use Sentence-BERT for semantic similarity
            old_embedding = self.sentence_bert_model.encode([old_semantic])[0]

            for element in current_screen_elements:
                element_semantic = self._get_external_semantic(element)
                if not element_semantic:
                    continue

                element_embedding = self.sentence_bert_model.encode([element_semantic])[0]

                # Cosine similarity
                similarity = self._cosine_similarity(old_embedding, element_embedding)

                if similarity > best_score:
                    best_score = similarity
                    best_match = element
        else:
            # Fallback: simple string matching
            for element in current_screen_elements:
                element_semantic = self._get_external_semantic(element)
                if not element_semantic:
                    continue

                similarity = self._string_similarity(old_semantic, element_semantic)

                if similarity > best_score:
                    best_score = similarity
                    best_match = element

        # Only return match if score exceeds threshold
        if best_score < self.external_threshold:
            best_match = None

        # Cache result
        self.match_cache[cache_key] = (best_match, best_score)

        return best_match, best_score

    def get_internal_semantic_match(self,
                                   old_element: Dict,
                                   all_updated_elements: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Find matching element based on internal semantics (code snippets).

        According to the paper (Section 3.4.1):
        - Uses fine-tuned UniXcoder for code clone detection
        - Searches all elements in updated version
        - Returns element with highest similarity score

        Args:
            old_element: GUI element from base version
            all_updated_elements: All GUI elements in updated version

        Returns:
            Tuple of (best matching element or None, similarity score)
        """
        old_code = old_element.get('internal_semantic', '')
        if not old_code:
            return None, 0.0

        best_match = None
        best_score = 0.0

        if self.use_unixcoder and self.unixcoder_model:
            # Use UniXcoder for code similarity
            for element in all_updated_elements:
                element_code = element.get('internal_semantic', '')
                if not element_code:
                    continue

                # UniXcoder would return similarity score
                similarity = self._code_similarity_unixcoder(old_code, element_code)

                if similarity > best_score:
                    best_score = similarity
                    best_match = element
        else:
            # Fallback: simple code similarity
            for element in all_updated_elements:
                element_code = element.get('internal_semantic', '')
                if not element_code:
                    continue

                similarity = self._code_similarity_simple(old_code, element_code)

                if similarity > best_score:
                    best_score = similarity
                    best_match = element

        return best_match, best_score

    def _get_external_semantic(self, element: Dict) -> str:
        """
        Extract external semantic from GUI element.

        According to the paper (Section 3.2.1):
        Priority order: text > content-desc > resource-id

        For elements without any of these properties (e.g., CheckBox),
        attempts to find the semantic from their perceptual group.

        Args:
            element: GUI element dictionary

        Returns:
            External semantic string
        """
        # Priority 1: text property
        text = element.get('text', '').strip()
        if text:
            return text

        # Priority 2: content-desc property
        content_desc = element.get('content-desc', '').strip()
        if content_desc:
            return content_desc

        # Priority 3: resource-id property
        resource_id = element.get('resource-id', '').strip()
        if resource_id:
            # Extract just the id name, not the full package
            if ':id/' in resource_id:
                return resource_id.split(':id/')[-1]
            return resource_id

        # Priority 4: Perceptual Group recognition (for CheckBox, etc.)
        # According to the paper, elements like CheckBox often don't have
        # meaningful properties and need to be associated with surrounding elements
        perceptual_semantic = self._get_perceptual_group_semantic(element)
        if perceptual_semantic:
            return perceptual_semantic

        return ''

    def _get_perceptual_group_semantic(self, element: Dict) -> str:
        """
        Get external semantic from perceptual group for elements without properties.

        According to the paper (Section 3.2.1):
        - Elements like CheckBox, SwitchButton, EditText often have empty properties
        - They form "perceptual groups" with a central element (usually TextView)
        - The central element has meaningful textual information
        - We assign the central element's semantic to the target element

        Implementation strategy (heuristic-based):
        1. Check if element is a type that typically needs perceptual group handling
        2. Find sibling elements (same parent container)
        3. Identify the central element (element with text in the same group)
        4. Return the central element's semantic

        Args:
            element: GUI element dictionary

        Returns:
            External semantic from the perceptual group's central element, or empty string
        """
        # Element types that typically need perceptual group recognition
        PERCEPTUAL_GROUP_TYPES = [
            'CheckBox', 'SwitchButton', 'Switch', 'EditText',
            'RadioButton', 'ToggleButton', 'ImageButton'
        ]

        # Check if this element needs perceptual group handling
        element_class = element.get('class', '')
        needs_perceptual_group = any(
            widget_type in element_class for widget_type in PERCEPTUAL_GROUP_TYPES
        )

        if not needs_perceptual_group:
            return ''

        # Try to find semantic from siblings
        # This is a heuristic approach: look for nearby elements with text
        siblings = element.get('siblings', [])
        if not siblings:
            # If siblings not provided, try to find from parent's children
            parent_children = element.get('parent_children', [])
            if parent_children:
                siblings = parent_children

        # Look for the central element: a sibling with meaningful text
        for sibling in siblings:
            # Skip self
            if sibling.get('id') == element.get('id'):
                continue

            # Check if this sibling is a TextView or similar text-holding element
            sibling_class = sibling.get('class', '')
            is_text_element = any(
                text_type in sibling_class
                for text_type in ['TextView', 'TextLabel', 'Label']
            )

            # If it's a text element or has text, it's likely the central element
            sibling_text = sibling.get('text', '').strip()
            if sibling_text and (is_text_element or len(sibling_text) > 2):
                # Found the central element with meaningful text
                return sibling_text

            # Also check content-desc of siblings
            sibling_desc = sibling.get('content-desc', '').strip()
            if sibling_desc and len(sibling_desc) > 2:
                return sibling_desc

        # Fallback: try to find semantic from spatial proximity
        # Look for elements with similar vertical position (same row)
        element_bounds = element.get('bounds', {})
        if element_bounds:
            element_y = element_bounds.get('y', 0)
            element_height = element_bounds.get('height', 0)

            # Check all siblings for vertical alignment
            for sibling in siblings:
                sibling_bounds = sibling.get('bounds', {})
                if not sibling_bounds:
                    continue

                sibling_y = sibling_bounds.get('y', 0)
                sibling_height = sibling_bounds.get('height', 0)

                # Check if vertically aligned (within threshold)
                y_diff = abs(element_y - sibling_y)
                if y_diff < max(element_height, sibling_height) * 0.5:
                    # Vertically aligned, check if has text
                    sibling_text = sibling.get('text', '').strip()
                    if sibling_text:
                        return sibling_text

        return ''

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _string_similarity(self, str1: str, str2: str) -> float:
        """
        Simple string similarity fallback.
        Uses Levenshtein distance ratio.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _code_similarity_simple(self, code1: str, code2: str) -> float:
        """
        Simple code similarity (fallback when UniXcoder not available).
        Uses token-based similarity.
        """
        # Tokenize code
        tokens1 = set(re.findall(r'\w+', code1))
        tokens2 = set(re.findall(r'\w+', code2))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _code_similarity_unixcoder(self, code1: str, code2: str) -> float:
        """
        Code similarity using UniXcoder model.
        Uses cosine similarity between code embeddings.
        """
        if not hasattr(self, 'unixcoder_model') or self.unixcoder_model is None:
            # Fallback if model not loaded
            return self._code_similarity_simple(code1, code2)

        try:
            import torch

            # Tokenize both code snippets
            inputs1 = self.unixcoder_tokenizer(
                code1,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.unixcoder_device)

            inputs2 = self.unixcoder_tokenizer(
                code2,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.unixcoder_device)

            # Get embeddings
            with torch.no_grad():
                outputs1 = self.unixcoder_model(**inputs1)
                outputs2 = self.unixcoder_model(**inputs2)

                # Use [CLS] token embedding (first token)
                embedding1 = outputs1.last_hidden_state[:, 0, :].squeeze()
                embedding2 = outputs2.last_hidden_state[:, 0, :].squeeze()

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()

            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            print(f"Warning: UniXcoder similarity failed: {e}")
            # Fallback to simple similarity
            return self._code_similarity_simple(code1, code2)

    def match_element(self,
                     old_element: Dict,
                     current_screen_elements: List[Dict],
                     all_updated_elements: List[Dict]) -> Optional[Dict]:
        """
        Main entry point for element matching.
        Follows Algorithm 1 from the paper (lines 7-15).

        Strategy:
        1. Try external semantic matching first
        2. If external matching fails (score < threshold), try internal matching

        Args:
            old_element: Element from base version
            current_screen_elements: Elements on current screen of updated version
            all_updated_elements: All elements in updated version

        Returns:
            Matched element or None
        """
        # Step 1: Try external semantic matching (Algorithm 1, line 7)
        matched_element, ex_score = self.get_external_semantic_match(
            old_element, current_screen_elements
        )

        # Step 2: If external matching succeeds, return it (Algorithm 1, line 8)
        if ex_score >= self.external_threshold and matched_element:
            return matched_element

        # Step 3: If external matching fails, try internal semantic matching (Algorithm 1, line 9)
        matched_element = self.get_internal_semantic_match(
            old_element, all_updated_elements
        )

        return matched_element

    def construct_locator(self, element: Dict, screen: Dict) -> str:
        """
        Construct a locator for the matched element.

        Args:
            element: The matched GUI element
            screen: The screen containing the element

        Returns:
            Locator string for the element
        """
        # Prefer resource-id if available
        resource_id = element.get('resource-id', '')
        if resource_id:
            return f"id:{resource_id}"

        # Then try accessibility-id
        content_desc = element.get('content-desc', '')
        if content_desc:
            return f"accessibility_id:{content_desc}"

        # Then try text
        text = element.get('text', '')
        if text:
            return f"text:{text}"

        # Last resort: xpath
        xpath = element.get('xpath', '')
        if xpath:
            return f"xpath:{xpath}"

        return ""


def main():
    """Example usage of CodeMatchHandler."""
    handler = CodeMatchHandler(external_threshold=0.7)
    handler.initialize_models()

    # Example elements
    old_element = {
        'text': 'Select all',
        'content-desc': '',
        'resource-id': 'android:id/title',
        'internal_semantic': 'selectAllCheckboxes();'
    }

    current_screen_elements = [
        {
            'text': 'Select All',
            'content-desc': '',
            'resource-id': 'android.networkmonitor:id/action_select_all'
        },
        {
            'text': 'More options',
            'content-desc': 'More options',
            'resource-id': ''
        }
    ]

    matched, score = handler.get_external_semantic_match(old_element, current_screen_elements)

    if matched:
        print(f"Matched element: {matched}")
        print(f"Similarity score: {score:.3f}")
    else:
        print("No match found")


if __name__ == '__main__':
    main()

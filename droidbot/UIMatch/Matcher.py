"""
Matching algorithm module for comparing elements between original UI and new version UI
"""

import os
import numpy as np
from lxml import etree as ET
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from .utils import compute_ssim, read_image, get_element_xpath, compute_xpath_similarity, compute_bounds_similarity, encode_image_to_base64, draw_element_on_image, draw_original_element_on_image, draw_replay_element_on_image
from .utils import get_encoded_image, openai_chat, get_find_result, get_component_no
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib


class Matcher:
    def __init__(self, original_png, original_tree, original_element, replay_png, replay_tree, logger, cross_page=False):
        """
        Initialize matcher
        
        Args:
            original_png: Screenshot path of the original UI
            original_xml_tree: XML tree object of the original UI
            original_element: Target element object in the original UI
            replay_png: Screenshot path of the new version UI
            replay_xml_tree: XML tree object of the new version UI
        """
        self.original_png = original_png
        self.original_tree = original_tree
        self.original_element = original_element
        self.replay_png = replay_png
        self.replay_tree = replay_tree
        root = self.replay_tree.getroot()
        _HindenWidgetFilter(root)


        # Candidate elements and scores for similarity matching
        self.candidates = []
        self.scores = []

        self.cross_page = cross_page

        # Weights for similarity matching
        if cross_page == False:
            self.alpha = 0.4 # Visual similarity weight
            self.beta = 0.3 # Attribute similarity weight
            self.gamma = 0.2 # Spatial similarity weight
        else:
            self.alpha = 0.5 # Visual similarity weight
            self.beta = 0.5 # Attribute similarity weight
            self.gamma = 0 # Spatial similarity weight

        # For large model matching
        self.top_n = 80
        # self.model_name = "gpt-4.1-mini"
        self.model_name = "gpt-5.1"
        self.model_type = "gpt"
        self.api_key = os.getenv("API_KEY")

        self.logger = logger
        self.app_name = None
    
    def matching(self, app_name=None, without_llm=False, without_rule=False, original_next_screen_summary=None) -> Dict[str, Any]:
        """
        Execute matching, trying three matching strategies by priority

        Ablation experiments:
        1. without_llm: Do not use LLM matching
        2. without_rule: Do not use rule-based filtering
        3. original_next_screen_summary: Summary of the next screen

        Returns:
            Matching result, containing match status, bounds, and other information
        """
        self.app_name = app_name

        # 0. Mark the original element for debugging
        original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
        original_img = read_image(self.original_png)
        marked_original_img = draw_original_element_on_image(original_img, original_bounds)
        original_output_filename = self.original_png.replace(".png", "_marked_original_element.png")
        marked_original_img.save(original_output_filename)

        if without_rule:
            # When this parameter is true, skip rule-based matching and filtering, and use all components as candidates
            # Need to compute similarity scores, otherwise _mark_candidates_on_image cannot mark candidate elements
            similarity_result = self.similarity_matching(without_rule=True)
            llm_result = self.llm_matching(original_next_screen_summary)
            return llm_result
        elif without_llm:

            # 1. First try exact matching
            exact_result = self.exact_matching()
            if exact_result["success"]:
                return exact_result

            # 2. If exact matching fails, try similarity matching
            similarity_result = self.similarity_matching()
            if similarity_result["success"]:
                return similarity_result

            # Return similarity max score
            if not self.scores:  # Boundary check
                return {"success": False}
            max_score = max(self.scores)
            return {
                "success": True,
                "matched_element": self.candidates[self.scores.index(max_score)],
                "matching_method": "similarity",
                "score": max_score
            }
        else: # Normal flow
            # 1. First try exact matching
            exact_result = self.exact_matching()
            if exact_result["success"]:
                return exact_result
            
            # 2. If exact matching fails, try similarity matching
            similarity_result = self.similarity_matching()
            if similarity_result["success"]:
                return similarity_result

            # 3. If similarity matching also fails, try LLM matching
            llm_result = self.llm_matching(original_next_screen_summary)
            return llm_result

    def exact_matching(self) -> Dict[str, Any]:
        """
        Exact matching strategy, matching elements through exact attribute and feature comparison.
        Tries content-desc, resource-id, and text attributes in sequence.
        Returns success if a unique match is found; tries the next attribute if multiple or no matches are found.

        Returns:
            Matching result, containing success and matched_element
        """
        root = self.replay_tree.getroot()

        def norm(v):
            # Normalize None or "" to None
            return v if v not in (None, "") else None

        def find_by_attr(attr_name, value, case_insensitive=False):
            """Find elements by attribute, returns (matched element, match count)"""
            if value is None:
                return None, 0
            matched = None
            count = 0
            for element in root.iter():
                cur_value = norm(element.attrib.get(attr_name))
                if cur_value is not None:
                    if case_insensitive:
                        is_match = cur_value.lower() == value.lower()
                    else:
                        is_match = cur_value == value
                    if is_match:
                        matched = element
                        count += 1
            return matched, count

        # 1. Exact match by content-desc
        content_desc = norm(self.original_element.attrib.get("content-desc"))
        matched_element, matched_times = find_by_attr("content-desc", content_desc, case_insensitive=True)
        if matched_element is not None and matched_times == 1:
            return {
                "success": True,
                "matched_element": matched_element,
                "matching_method": "exact_content_desc"
            }

        # 2. Exact match by resource-id
        resource_id = norm(self.original_element.attrib.get("resource-id"))
        matched_element, matched_times = find_by_attr("resource-id", resource_id, case_insensitive=False)
        if matched_element is not None and matched_times == 1:
            return {
                "success": True,
                "matched_element": matched_element,
                "matching_method": "exact_resource_id"
            }

        # 3. Exact match by text
        text = norm(self.original_element.attrib.get("text"))
        matched_element, matched_times = find_by_attr("text", text, case_insensitive=True)
        if matched_element is not None and matched_times == 1:
            return {
                "success": True,
                "matched_element": matched_element,
                "matching_method": "exact_text"
            }

        # No unique match found
        return {
            "success": False
        }
    
    
    def similarity_matching(self, without_rule=False) -> Dict[str, Any]:
        """
        Similarity matching strategy, finding the best match through multi-dimensional similarity comparison

        1. Visual similarity
        2. Structural similarity
        3. Spatial similarity

        """
        self.candidates = self._candidate_elements()
        if len(self.candidates) == 0:
            return {
                "success": False
            }
        self.scores = []
        self.all_matched_elements = []
        for candidate in self.candidates:
            score1 = self._compute_visual_similarity(candidate)
            if score1 > 0.95:
                self.all_matched_elements.append(candidate)
            
            # score2 = self._compute_structure_similarity(candidate)
            
            # score3 = self._compute_space_similarity(candidate)
            
            score4 = self._compute_attribute_similarity(candidate)
            # Visual > Text > Spatial > Structural
            # total_score = self.alpha * score1 + self.beta * score4 + self.gamma * score3 + (1 - self.alpha - self.beta - self.gamma) * score2
            # Visual + Attribute
            total_score = 0.5 * score1 + 0.5 * score4
            self.scores.append(total_score)

        if len(self.all_matched_elements) == 1 and not without_rule:
            return {
                "success": True,
                "matched_element": self.all_matched_elements[0],
                "matching_method": "visual_similarity"
            }
        
        return {
            "success": False
        }

    
    
    def guider_matching(self) -> Dict[str, Any]:
        """
        Guider baseline matching strategy (ISSTA 2021)

        Implements three types of widget matching:
        - α-typed (Sure match): Unique identity property match (only one element matches)
        - β-typed (Close match): Multiple identity matches, ranked by visual similarity
        - γ-typed (Remote match): No identity match, only visual similarity

        Identity properties: resource-id, content-desc, text
        """
        root = self.replay_tree.getroot()

        def norm(v):
            return v if v not in (None, "") else None

        def find_all_by_attr(attr_name, value, case_insensitive=False):
            """Find all elements matching the given attribute"""
            if value is None:
                return []
            matched = []
            for element in root.iter():
                cur_value = norm(element.attrib.get(attr_name))
                if cur_value is not None:
                    if case_insensitive:
                        is_match = cur_value.lower() == value.lower()
                    else:
                        is_match = cur_value == value
                    if is_match:
                        matched.append(element)
            return matched

        # Get identity properties of original element
        content_desc = norm(self.original_element.attrib.get("content-desc"))
        resource_id = norm(self.original_element.attrib.get("resource-id"))
        text = norm(self.original_element.attrib.get("text"))

        # Log: output the identity properties of the original element
        self.logger.info(f"[Guider] === Original element identity properties ===")
        self.logger.info(f"[Guider]   content-desc: {content_desc}")
        self.logger.info(f"[Guider]   resource-id: {resource_id}")
        self.logger.info(f"[Guider]   text: {text}")
        original_bounds = self.original_element.attrib.get("bounds", "")
        self.logger.info(f"[Guider]   bounds: {original_bounds}")

        # === α-typed matching (Sure match) ===
        # Unique identity property match - only one element on the page matches
        self.logger.info(f"[Guider] === α-typed matching (Sure match) ===")

        # Debug: list all elements with a text attribute on the current page
        all_texts = []
        for element in root.iter():
            elem_text = norm(element.attrib.get("text"))
            if elem_text:
                all_texts.append(elem_text)
        self.logger.info(f"[Guider] [DEBUG] All texts in current page: {all_texts[:20]}{'...' if len(all_texts) > 20 else ''}")

        # Check content-desc
        if content_desc:
            matches = find_all_by_attr("content-desc", content_desc, case_insensitive=True)
            self.logger.info(f"[Guider]   content-desc '{content_desc}' matches: {len(matches)}")
            if len(matches) == 1:
                self.logger.info(f"[Guider]   ✓ α-typed match by content-desc!")
                return {
                    "success": True,
                    "matched_element": matches[0],
                    "matching_method": "guider_alpha_content_desc"
                }

        # Check resource-id
        if resource_id:
            matches = find_all_by_attr("resource-id", resource_id, case_insensitive=False)
            self.logger.info(f"[Guider]   resource-id '{resource_id}' matches: {len(matches)}")
            if len(matches) == 1:
                self.logger.info(f"[Guider]   ✓ α-typed match by resource-id!")
                return {
                    "success": True,
                    "matched_element": matches[0],
                    "matching_method": "guider_alpha_resource_id"
                }

        # Check text
        if text:
            matches = find_all_by_attr("text", text, case_insensitive=True)
            self.logger.info(f"[Guider]   text '{text}' matches: {len(matches)}")
            if len(matches) == 1:
                self.logger.info(f"[Guider]   ✓ α-typed match by text!")
                return {
                    "success": True,
                    "matched_element": matches[0],
                    "matching_method": "guider_alpha_text"
                }

        self.logger.info(f"[Guider]   No α-typed match found")

        # === β-typed matching (Close match) ===
        # Multiple widgets share identity property, rank by visual similarity
        self.logger.info(f"[Guider] === β-typed matching (Close match) ===")
        beta_candidates = []

        if content_desc:
            matches = find_all_by_attr("content-desc", content_desc, case_insensitive=True)
            if len(matches) > 1:
                self.logger.info(f"[Guider]   content-desc '{content_desc}' has {len(matches)} matches (>1)")
                beta_candidates.extend(matches)

        if resource_id:
            matches = find_all_by_attr("resource-id", resource_id, case_insensitive=False)
            if len(matches) > 1:
                self.logger.info(f"[Guider]   resource-id '{resource_id}' has {len(matches)} matches (>1)")
                beta_candidates.extend(matches)

        if text:
            matches = find_all_by_attr("text", text, case_insensitive=True)
            if len(matches) > 1:
                self.logger.info(f"[Guider]   text '{text}' has {len(matches)} matches (>1)")
                beta_candidates.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_beta_candidates = []
        for elem in beta_candidates:
            elem_id = id(elem)
            if elem_id not in seen:
                seen.add(elem_id)
                unique_beta_candidates.append(elem)

        self.logger.info(f"[Guider]   Total β-typed candidates: {len(unique_beta_candidates)}")

        if unique_beta_candidates:
            # Rank by visual similarity (SIFT as in Guider paper)
            scored_candidates = []
            for candidate in unique_beta_candidates:
                visual_score = self._compute_visual_similarity_sift(candidate)
                candidate_text = candidate.attrib.get("text", "")
                candidate_desc = candidate.attrib.get("content-desc", "")
                candidate_bounds = candidate.attrib.get("bounds", "")
                self.logger.info(f"[Guider]   β candidate: text='{candidate_text}', desc='{candidate_desc}', bounds={candidate_bounds}, SIFT={visual_score:.4f}")
                scored_candidates.append((candidate, visual_score))

            # Sort by visual similarity descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            if scored_candidates:
                best_candidate, best_score = scored_candidates[0]
                best_text = best_candidate.attrib.get("text", "")
                best_desc = best_candidate.attrib.get("content-desc", "")
                best_bounds = best_candidate.attrib.get("bounds", "")
                self.logger.info(f"[Guider]   ✓ β-typed match! Best: text='{best_text}', desc='{best_desc}', bounds={best_bounds}, SIFT={best_score:.4f}")
                return {
                    "success": True,
                    "matched_element": best_candidate,
                    "matching_method": "guider_beta",
                    "visual_similarity": best_score
                }
        else:
            self.logger.info(f"[Guider]   No β-typed candidates found")

        # === γ-typed matching (Remote match) ===
        # No identity property match, only visual similarity
        self.logger.info(f"[Guider] === γ-typed matching (Remote match) ===")
        all_candidates = self._candidate_elements(without_rule=True)
        self.logger.info(f"[Guider]   Total candidates before filtering: {len(all_candidates) if all_candidates else 0}")

        if all_candidates:
            scored_candidates = []
            skipped_empty_bounds = 0
            skipped_non_leaf = 0
            skipped_status_bar = 0
            for candidate in all_candidates:
                bounds = candidate.attrib.get("bounds", "")
                # Filter out elements with empty bounds
                if not bounds or not self._parse_bounds(bounds):
                    skipped_empty_bounds += 1
                    continue
                # Filter out non-leaf nodes (nodes with child elements)
                if len(list(candidate)) > 0:
                    skipped_non_leaf += 1
                    continue
                # Filter out elements in the status bar area (y < 70 is usually the system status bar, not part of the app)
                parsed_bounds = self._parse_bounds(bounds)
                if parsed_bounds and parsed_bounds[1] < 70:  # top y < 70
                    skipped_status_bar += 1
                    continue
                visual_score = self._compute_visual_similarity_sift(candidate)
                scored_candidates.append((candidate, visual_score))

            self.logger.info(f"[Guider]   Skipped {skipped_empty_bounds} candidates with empty bounds")
            self.logger.info(f"[Guider]   Skipped {skipped_non_leaf} candidates (non-leaf nodes)")
            self.logger.info(f"[Guider]   Skipped {skipped_status_bar} candidates (status bar area)")
            self.logger.info(f"[Guider]   Total γ-typed leaf candidates: {len(scored_candidates)}")

            # Sort by visual similarity descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Output detailed info for the top 10 candidates
            self.logger.info(f"[Guider]   === Top 10 γ-typed candidates by SIFT ===")
            for i, (cand, score) in enumerate(scored_candidates[:10]):
                cand_text = cand.attrib.get("text", "")
                cand_desc = cand.attrib.get("content-desc", "")
                cand_class = cand.attrib.get("class", "")
                cand_bounds = cand.attrib.get("bounds", "")
                self.logger.info(f"[Guider]   #{i+1}: SIFT={score:.4f}, text='{cand_text}', desc='{cand_desc}', class='{cand_class}', bounds={cand_bounds}")

            # SIFT threshold: values below this are not considered valid matches
            SIFT_THRESHOLD = 0.15

            if scored_candidates and scored_candidates[0][1] >= SIFT_THRESHOLD:
                best_candidate, best_score = scored_candidates[0]
                best_text = best_candidate.attrib.get("text", "")
                best_desc = best_candidate.attrib.get("content-desc", "")
                best_class = best_candidate.attrib.get("class", "")
                best_bounds = best_candidate.attrib.get("bounds", "")
                self.logger.info(f"[Guider]   ✓ γ-typed match! Best: text='{best_text}', desc='{best_desc}', class='{best_class}', bounds={best_bounds}, SIFT={best_score:.4f}")
                return {
                    "success": True,
                    "matched_element": best_candidate,
                    "matching_method": "guider_gamma",
                    "visual_similarity": best_score
                }
            else:
                if scored_candidates:
                    self.logger.info(f"[Guider]   No γ-typed match found (best score {scored_candidates[0][1]:.4f} < threshold {SIFT_THRESHOLD})")
                else:
                    self.logger.info(f"[Guider]   No γ-typed match found (no candidates)")
        else:
            self.logger.info(f"[Guider]   No candidates available for γ-typed matching")

        self.logger.info(f"[Guider] === Matching failed ===")
        return {
            "success": False
        }
    
    def _compute_attribute_similarity(self, candidate_element) -> float:
        """Compute attribute similarity: text"""
        original_text = self.original_element.attrib.get("text")
        candidate_text = candidate_element.attrib.get("text")
        original_content_desc = self.original_element.attrib.get("content-desc")
        candidate_content_desc = candidate_element.attrib.get("content-desc")
        original_resource_id = self.original_element.attrib.get("resource-id")
        candidate_resource_id = candidate_element.attrib.get("resource-id")
        original_str = f"{original_text}|{original_content_desc}|{original_resource_id}"
        candidate_str = f"{candidate_text}|{candidate_content_desc}|{candidate_resource_id}"
        seq_matcher = difflib.SequenceMatcher(None, original_str.lower().split("|"), candidate_str.lower().split("|"))
        return round(seq_matcher.ratio(), 4)

    def _compute_text_attr_similarity(self, candidate_element) -> float:
        """
        Compute text attribute similarity: content-desc, text, class, resource-id
        Used to select the best element when multiple candidate elements are all different
        """
        def safe_str(val):
            return str(val).lower() if val else ""

        original_text = safe_str(self.original_element.attrib.get("text"))
        candidate_text = safe_str(candidate_element.attrib.get("text"))
        original_content_desc = safe_str(self.original_element.attrib.get("content-desc"))
        candidate_content_desc = safe_str(candidate_element.attrib.get("content-desc"))
        original_class = safe_str(self.original_element.attrib.get("class"))
        candidate_class = safe_str(candidate_element.attrib.get("class"))
        original_resource_id = safe_str(self.original_element.attrib.get("resource-id"))
        candidate_resource_id = safe_str(candidate_element.attrib.get("resource-id"))

        # Compute similarity for each attribute
        text_sim = difflib.SequenceMatcher(None, original_text, candidate_text).ratio() if original_text or candidate_text else 1.0
        desc_sim = difflib.SequenceMatcher(None, original_content_desc, candidate_content_desc).ratio() if original_content_desc or candidate_content_desc else 1.0
        class_sim = 1.0 if original_class == candidate_class else 0.0
        resource_id_sim = difflib.SequenceMatcher(None, original_resource_id, candidate_resource_id).ratio() if original_resource_id or candidate_resource_id else 1.0

        # Weighted average
        return round(0.3 * text_sim + 0.3 * desc_sim + 0.1 * class_sim + 0.3 * resource_id_sim, 4)


    def _candidate_elements(self, without_rule=False) -> List[ET.Element]:
        """
        Find possible candidate elements and filter them based on a set of rules

        Filtering rules:
        1. Keep elements with visible=true, remove invisible elements
        2. For parent-child boundary overlap >= 0.95, keep only the leaf node
        3. Remove extremely small elements (area of 2-3 pixels)
        4. Remove elements whose area is more than 5 times that of the original element
        5. Filter out system UI elements (status bar and navigation bar)

        Returns:
            Filtered list of candidate elements
        """


        # Get size information of the original element
        original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
        if not original_bounds:
            print("Unable to parse bounds of the original element")
            return []
                    
        # Initial candidate list - get all node elements
        root = self.replay_tree.getroot()

        initial_candidates = list(root.iter()) # Get all elements
        print("initial_candidates", len(initial_candidates))

        if without_rule:
            return initial_candidates
        
        # Apply filtering rules
        filtered_candidates = []

        
        for element in initial_candidates:
            if element.attrib.get("covered", "false") == "true" and self.app_name is not None and self.app_name != "com.appmindlab.nano": # Already covered by other elements
                continue

            # Rule 1: Keep only visible elements
            if element.attrib.get("visible-to-user", "true").lower() == "false":
                continue
            
            # Rule 5: Filter out system UI elements (status bar and navigation bar)
            package = element.attrib.get("package", "")
            if package == "com.android.systemui":
                continue
                
            # Rule 2: Parse the element's bounds
            bounds = self._parse_bounds(element.attrib.get("bounds", ""))
            if not bounds:
                continue
                
            # Rule 3: Filter out extremely small elements
            area = self._calculate_area(bounds)
            if area < 10:  # Filter out elements with area less than 10 pixels
                continue
                
            # Rule 4: Filter out oversized elements
            # if area > original_area * 5:
            #     continue

            # Rule 6: Filter out elements with width < 5 or height < 5
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if width < 10 or height < 10:
                continue
                
            # Rule 5: For high parent-child boundary overlap, keep only the leaf node
            is_leaf_or_unique = True
            
            # Check if the element has child elements with high boundary overlap
            for child in element:
                child_bounds = self._parse_bounds(child.attrib.get("bounds", ""))
                if child_bounds:
                    overlap_ratio = self._calculate_overlap(bounds, child_bounds) / area if area > 0 else 0
                    if overlap_ratio > 0.95:  # Boundary overlap exceeds 95%
                        is_leaf_or_unique = False
                        break
            
            if is_leaf_or_unique:
                filtered_candidates.append(element)
        
        leaf_candidates = []
        for candidate in filtered_candidates:
            if len(candidate) > 0: # leaf node
                continue
            leaf_candidates.append(candidate)

        print("filtered_candidates", len(filtered_candidates))
        print("leaf_candidates", len(leaf_candidates))
        
        if len(leaf_candidates) == 0:
            return filtered_candidates
        else:
            return leaf_candidates
        
    def _parse_bounds(self, bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Parse the bounds string of an Android UI element, format: [x1,y1][x2,y2]

        Args:
            bounds_str: bounds attribute string

        Returns:
            (left, top, right, bottom) tuple, or None if parsing fails
        """
        try:
            # Extract coordinate values
            import re
            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"Failed to parse bounds: {bounds_str}, error: {e}")
        return None
        
    def _calculate_area(self, bounds: Tuple[int, int, int, int]) -> int:
        """
        Calculate the area of an element

        Args:
            bounds: (left, top, right, bottom) tuple

        Returns:
            Area of the element (in pixels)
        """
        left, top, right, bottom = bounds
        width = max(0, right - left)
        height = max(0, bottom - top)
        return width * height
        
    def _calculate_overlap(self, bounds1: Tuple[int, int, int, int], 
                          bounds2: Tuple[int, int, int, int]) -> int:
        """
        Calculate the overlap area between two element boundaries

        Args:
            bounds1: (left, top, right, bottom) of the first element
            bounds2: (left, top, right, bottom) of the second element

        Returns:
            Area of the overlapping region
        """
        left1, top1, right1, bottom1 = bounds1
        left2, top2, right2, bottom2 = bounds2
        
        # Calculate the overlap region
        overlap_left = max(left1, left2)
        overlap_top = max(top1, top2)
        overlap_right = min(right1, right2)
        overlap_bottom = min(bottom1, bottom2)
        
        # Check if there is overlap
        if overlap_right > overlap_left and overlap_bottom > overlap_top:
            return (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
        return 0
    
    def _compute_visual_similarity(self, candidate_element) -> float:
        """
        Calculate the visual similarity between the original element and the candidate element on the image.
        Uses SSIM (Structural Similarity Index) for image similarity computation.

        Args:
            candidate_element: Candidate element

        Returns:
            Visual similarity score (0.0-1.0)
        """
        try:
            # 1. Get bounds of the original and candidate elements
            original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
            candidate_bounds = self._parse_bounds(candidate_element.attrib.get("bounds", ""))

            # 2. Read images
            original_img = read_image(self.original_png)
            replay_img = read_image(self.replay_png)

            # 3. Crop element images
            original_x1, original_y1, original_x2, original_y2 = original_bounds
            candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate_bounds
            
            # Use PIL's crop method to crop the image
            try:
                original_crop = original_img.crop((original_x1, original_y1, original_x2, original_y2))
                candidate_crop = replay_img.crop((candidate_x1, candidate_y1, candidate_x2, candidate_y2))
                
                # Check if the cropped image is empty
                if original_crop.width <= 1 or original_crop.height <= 1 or candidate_crop.width <= 1 or candidate_crop.height <= 1:
                    return 0.0
            except Exception as e:
                print(f"Image cropping error: {e}")
                return 0.0

            similarity = compute_ssim(original_crop, candidate_crop)
            
            return similarity
            
        except Exception as e:
            print(f"Error computing visual similarity: {e}")
            return 0.0

    def _compute_visual_similarity_sift(self, candidate_element) -> float:
        """
        Calculate visual similarity using SIFT feature matching (Guider ISSTA 2021 paper method)

        Args:
            candidate_element: Candidate element

        Returns:
            Visual similarity score (0.0-1.0)
        """
        try:
            import cv2

            # 1. Get bounds of the original and candidate elements
            original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
            candidate_bounds = self._parse_bounds(candidate_element.attrib.get("bounds", ""))

            # 2. Read images
            original_img = read_image(self.original_png)
            replay_img = read_image(self.replay_png)

            # 3. Crop element images
            original_x1, original_y1, original_x2, original_y2 = original_bounds
            candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate_bounds

            try:
                original_crop = original_img.crop((original_x1, original_y1, original_x2, original_y2))
                candidate_crop = replay_img.crop((candidate_x1, candidate_y1, candidate_x2, candidate_y2))

                # Check if the cropped image is empty
                if original_crop.width <= 1 or original_crop.height <= 1 or candidate_crop.width <= 1 or candidate_crop.height <= 1:
                    return 0.0
            except Exception as e:
                print(f"Image cropping error: {e}")
                return 0.0

            # 4. Convert to OpenCV format (grayscale)
            original_cv = cv2.cvtColor(np.array(original_crop), cv2.COLOR_RGB2GRAY)
            candidate_cv = cv2.cvtColor(np.array(candidate_crop), cv2.COLOR_RGB2GRAY)

            # 5. Create SIFT detector
            sift = cv2.SIFT_create()

            # 6. Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(original_cv, None)
            kp2, des2 = sift.detectAndCompute(candidate_cv, None)

            # If no feature points detected, return 0
            if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                return 0.0

            # 7. Use BFMatcher for feature matching
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 8. Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # 9. Calculate similarity score
            # Use the ratio of good matches to the number of original feature points
            if len(kp1) == 0:
                return 0.0

            similarity = len(good_matches) / max(len(kp1), len(kp2))
            # Normalize to 0-1
            similarity = min(1.0, similarity)

            return similarity

        except Exception as e:
            print(f"Error computing SIFT visual similarity: {e}")
            return 0.0

    def _compute_structure_similarity(self, candidate_element) -> float:
        """Compute structural similarity, the similarity between two xpaths"""
        # 1. Get the xpath of the original and candidate elements
        original_xpath = get_element_xpath(self.original_tree, self.original_element)
        candidate_xpath = get_element_xpath(self.replay_tree, candidate_element)
        
        # 2. Compute the similarity between two xpaths
        similarity = compute_xpath_similarity(original_xpath, candidate_xpath)
        return similarity


    def _compute_space_similarity(self, candidate_element) -> float:
        """Compute spatial similarity"""
        original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
        candidate_bounds = self._parse_bounds(candidate_element.attrib.get("bounds", ""))
        
        # 2. Compute the similarity between two bounds
        similarity = compute_bounds_similarity(original_bounds, candidate_bounds)
        return similarity

  
  
    def llm_matching(self, original_next_screen_summary=None) -> Dict[str, Any]:
        """
        LLM matching strategy, using a large language model for UI element matching

        Returns:
            Matching result, containing success and matched_element
        """
        if len(self.candidates) == 0:
            return {
                "success": False
            }

        # 1. Mark candidate elements
        marked_replay_img = self._mark_candidates_on_image()
        output_filename = self.replay_png.replace(".png", "_marked_candidates.png") # Stored alongside the original
        marked_replay_img.save(output_filename)
        
        # 2. Mark the original element
        original_bounds = self._parse_bounds(self.original_element.attrib.get("bounds", ""))
        original_img = read_image(self.original_png)
        marked_original_img = draw_original_element_on_image(original_img, original_bounds)
        original_output_filename = self.original_png.replace(".png", "_marked_original_element.png")
        marked_original_img.save(original_output_filename)

        # 3. Image of the original element
        original_element_img = original_img.crop(original_bounds)
        # Save for debugging
        original_element_img_filename = self.original_png.replace(".png", "_original_element.png")
        original_element_img.save(original_element_img_filename)
        
        # 4. Determine if the page is related
        # if self.cross_page == False:
        #     found_page_related = self._check_page_found(marked_original_img, original_element_img, marked_replay_img)
        #     self.logger.info(f"found_page_related: {found_page_related}")
        #     if found_page_related == "NO":
        #         return {
        #             "success": False
        #         }

        # 4. If the page is related, find the element
        self.logger.info(f"original_png: {self.original_png}")
        try_times = 3
        element_id = None
        candidate_elements = []

        # Execute LLM calls concurrently
        with ThreadPoolExecutor(max_workers=try_times) as executor:
            futures = [
                executor.submit(self._find_element, marked_original_img, original_element_img, marked_replay_img, original_next_screen_summary)
                for _ in range(try_times)
            ]
            for future in as_completed(futures):
                candidate_element_id = future.result()
                if candidate_element_id is not None:
                    candidate_elements.append(candidate_element_id)

        # for _ in range(try_times):
        #     candidate_element_id = self._find_element(marked_original_img, original_element_img, marked_replay_img)
        #     if candidate_element_id is not None:
        #         candidate_elements.append(candidate_element_id)

        # voting for the best element
        if len(candidate_elements) == try_times:
            total_candidates = set(candidate_elements)
            if len(total_candidates) == 1:
                element_id = list(total_candidates)[0]
            elif len(total_candidates) == 3:
                # All 3 elements are different, call LLM 2 more times, then vote using the 5 results
                self.logger.info("All 3 LLM results are different, calling LLM 2 more times for 5-way voting...")
                additional_tries = 2
                with ThreadPoolExecutor(max_workers=additional_tries) as executor:
                    futures = [
                        executor.submit(self._find_element, marked_original_img, original_element_img, marked_replay_img, original_next_screen_summary)
                        for _ in range(additional_tries)
                    ]
                    for future in as_completed(futures):
                        candidate_element_id = future.result()
                        if candidate_element_id is not None:
                            candidate_elements.append(candidate_element_id)

                self.logger.info(f"5-way voting candidates: {candidate_elements}")
                element_id = self._voting_for_best_element(candidate_elements)
            else:
                element_id = self._voting_for_best_element(candidate_elements)

        
        if element_id is not None and element_id>=0 and element_id<len(self.candidates):
            found_element = self.candidates[int(element_id)]
            # corresponding_score = self.scores[int(element_id)]
            # if corresponding_score < 0.4:
            #     return {
            #         "success": False
            #     }
            #     return self._rollback_matching()
            # else:
            # print(f"found_element: {self.scores[int(element_id)]}")

            
            return {
                "success": True,
                "matched_element": found_element,
                "matching_method": "llm"
            }

        return {
            "success": False
        }

    def _voting_for_best_element(self, ids) -> int:
        
        return Counter(ids).most_common(1)[0][0]

        

    def _rollback_matching(self) -> Dict[str, Any]:
        """
        Rollback matching, return the one with the highest score
        """
        # Return the one with the highest score
        best_score = max(self.scores)
        if best_score < 0.7:
            return {
                "success": False
            }
        best_index = self.scores.index(best_score)
        best_candidate = self.candidates[best_index]
        return {
            "success": True,
            "matched_element": best_candidate,
            "matching_method": "hybrid_similarity",
            "score": best_score
        }
    
    def _find_element(self, marked_original_img, original_element_img, marked_replay_img, original_next_screen_summary=None) -> Dict[str, Any]:
        """
        Find element
        """
        # 1. encode image to base64
        marked_original_img_base64 = get_encoded_image(marked_original_img)
        marked_replay_img_base64 = get_encoded_image(marked_replay_img)
        original_element_img_base64 = get_encoded_image(original_element_img)

        # 2. construct prompt
        system_prompt, user_prompt = self._construct_find_element_llm_prompt(marked_original_img_base64, original_element_img_base64, marked_replay_img_base64, original_next_screen_summary)

        # 3. call llm
        if self.model_type == "gpt" or self.model_type == "deepseek":
            response, token_usage = openai_chat(system_prompt, user_prompt, self.api_key, self.model_name, self.model_type)
            self.logger.info(f"response: {response}")
            self.logger.info(f"token_usage: {token_usage}")
            
            # Parse the component ID returned by LLM
            element_id_str = get_component_no(response)
            self.logger.info(f"Component ID returned by LLM: {element_id_str}")
            
            # Handle possible multiple ID case
            try:
                # If multiple IDs were returned (e.g., "2, 3, 4"), take the first one
                if ',' in element_id_str:
                    element_ids = [id.strip() for id in element_id_str.split(',')]
                    self.logger.info(f"LLM returned multiple IDs: {element_ids}, will use the first one: {element_ids[0]}")
                    element_id = int(element_ids[0])
                else:
                    element_id = int(element_id_str)
                
                return element_id
            except (ValueError, IndexError) as e:
                self.logger.error(f"Failed to parse component ID: {e}, original response: {element_id_str}")
                # Return None to indicate parsing failure
                return None
        else:
            self.logger.error(f"Unsupported model type: {self.model_type}")
            return None

    def _check_page_found(self, marked_original_img, original_element_img, marked_replay_img) -> bool:
        """
        Determine if the page is related

        Args:
            original_img: Original element image object
            replay_img: Image object with marked candidate elements
        """
        # 1. encode image to base64
        marked_original_img_base64 = get_encoded_image(marked_original_img)
        marked_replay_img_base64 = get_encoded_image(marked_replay_img)
        original_element_img_base64 = get_encoded_image(original_element_img)

        # 2. construct prompt
        system_prompt, user_prompt = self._construct_page_found_llm_prompt(marked_original_img_base64, original_element_img_base64, marked_replay_img_base64)
        
        # 3. call llm
        if self.model_type == "gpt" or self.model_type == "deepseek":
            response, token_usage = openai_chat(system_prompt, user_prompt, self.api_key, self.model_name, self.model_type)
            find_result = get_find_result(response)
            return find_result
        else:
            print("Unsupported model type")
            return False
        
    
    def _mark_candidates_on_image(self) -> Image.Image:
        """
        Mark candidate elements on the new version UI screenshot

        Returns:
            (marked_image_path, marked_components_dict): Marked image path and component dictionary
        """
        if len(self.candidates) == 0: # If the candidate list is empty, perform similarity matching to initialize self.candidates and self.scores
            self.similarity_matching()

        # Create a temporary image path
        replay_img = read_image(self.replay_png)
        
        # Create a list of (index, score) pairs
        indexed_scores = [(i, score) for i, score in enumerate(self.scores)]
        
        # Sort by score in descending order
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get the indices of the top N highest-scoring candidate elements
        top_n = min(self.top_n, len(indexed_scores))  # Mark at most self.top_n elements
        top_n_candidates_indexes = [idx for idx, _ in indexed_scores[:top_n]]

        for i in range(len(self.candidates)):
            if i in top_n_candidates_indexes:
                element = self.candidates[i]
                bounds_str = element.attrib.get("bounds", "")
                if not bounds_str:
                    continue
                replay_img = draw_replay_element_on_image(replay_img, bounds_str, id=i)

        return replay_img
            
    
    def _construct_page_found_llm_prompt(self, marked_original_img_base64, original_element_img_base64, marked_replay_img_base64) -> Tuple[str, Dict]:
        """
        Construct LLM prompt
        """
        analyze_ori_scenarios_prompt = f"""
I will provide you with the original application version's screenshot (marked with red boxes indicating an original UI component) and the original UI component Figure.


* Original Screenshot
```
please see the Figure 1.
```

* Original UI component Figure
```
please see the UI component figure.
```
"""
    
        analyze_update_effects_prompt = f"""
I will provide you with the updated application version's screenshot, Please analyze whether the marked components in the old version screenshot can be found in the new version screenshot.

* Updated Screenshot
```
please see the Figure 2.
```
"""
    
        system_prompt = """
You are an Android UI testing expert.

Your task is to determine whether the UPDATED screen represents the SAME **PAGE-LEVEL INTENT / FUNCTIONAL STAGE** as the ORIGINAL screen — not whether specific UI components or content appear identical.

This task focuses on **functional relevance**, not exact UI matching.

---

## What “Page Found” Means (IMPORTANT)

A page should be considered FOUND (YES) if:
- The UPDATED screen serves the SAME PURPOSE as the ORIGINAL screen
- The user has reached the SAME FUNCTIONAL STAGE in the workflow
- A tester would consider the navigation successful for continuing the same test case

---

## Strong YES Rules (Based on Testing Practice)

You should IMMEDIATELY return YES if **any** of the following holds:

1. **Identical or functionally equivalent core UI components are present**
   - Even if surrounding content differs
   - Even if displayed text or data is different

2. **The page is a LIST-type or FEED-type page**
   - Examples: news list, article list, item list, browsing pages
   - Ignore the specific list items or content
   - If the page clearly supports the same browsing/reading purpose, return YES

These cases indicate the tester has successfully reached the intended page,
regardless of content differences.

---

## Focus on PAGE INTENT, not CONTENT DETAILS

You MUST focus on the **INTENT and PURPOSE of the page**, for example:
- Reading a news article
- Browsing a list of news or items
- Viewing content details
- Accessing a feature page
- Performing a configuration or settings task

IGNORE:
- Differences in displayed content (e.g., different news text)
- Differences between individual list items
- Visual style changes
- UI restructuring
- Minor layout or navigation differences

---

## What Does NOT Count as Page Found (NO)

Return NO only if:
- The UPDATED screen represents a DIFFERENT functional stage
- The user is on an unrelated feature or page
- The navigation failed to reach the intended page purpose

---

## Your Task

1. Analyze the ORIGINAL screen and infer its **page-level intent**.
2. Analyze the UPDATED screen and infer its **page-level intent**.
3. Decide whether both screens represent the SAME intent / purpose.

EXAMPLE OUTPUT:

```result.md
### Analyze_Process  
Analyze the UI-related info between original and updated version, and explain whether the marked components in the old version screenshot can be found in the new version screenshot..

### Your Answer
YES or NO
```

"""

        user_prompt = {}

        user_prompt['ori_analyze'] = [marked_original_img_base64] + [original_element_img_base64] + [
            {"type": "text", "text": analyze_ori_scenarios_prompt}
        ]

        user_prompt['update_analyze'] = [marked_replay_img_base64] + [
            {"type": "text", "text": analyze_update_effects_prompt}
        ]

        return system_prompt, user_prompt
    
    def _extract_component_no(self, text: str) -> str:
        """
        Extract component number from LLM response

        Args:
            text: LLM response text

        Returns:
            Component number
        """
        import re
        
        # Prefer matching the content inside brackets on the line after "### Matched_UI_No"
        match = re.search(r'### Matched_UI_No\s*\n\s*\[(.*?)\]', text)
        
        if match:
            result = match.group(1)
            return result
        else:
            # Fallback: find the content inside the last []
            matches = re.findall(r'\[(.*?)\]', text)
            if matches:
                result = matches[-1]  # Take the last one
                return result
        
        return ""

#     def _construct_find_element_llm_prompt(self, marked_original_img_base64, original_element_img_base64, marked_replay_img_base64) -> Tuple[str, Dict]:
#         """
#         Construct LLM prompt
#         """

#         analyze_ori_scenarios_prompt = f"""
# I will provide you with the original application version's screenshot (marked with red boxes indicating an original UI component) and the original UI component Figure.


# * Original Screenshot
# ```
# please see the Figure 1.
# ```

# * Original UI component Figure
# ```
# please see the UI component figure.
# ```
# """
    
#         analyze_update_effects_prompt = f"""
# I will provide you with the updated application version's screenshot. Different UI components are marked with green boxes and assigned a numerical sequence number. You need to analyze the screenshots of the Updated version and identify an UI component that is most similar to the original UI component of the Original version, then provide the component sequence numbers.

# * Updated Screenshot
# ```
# please see the Figure 2.
# ```
# """
    
#         system_prompt = """
# You are an Android developer who is skilled at analyzing component location relationships by combining GUI.

# There is a special task scenario that you need to solve. In the software version iteration, an UI component of the updated version may change compared to the original version. You need to browse the original version of the UI component information, and then analyze the updated version of the UI component information to find the best matched UI number in the Updated Screenshot.

# First, you will obtain the original version of the UI component information, including screenshots of the original component (marked with red boxes indicating the original UI component).
# Second, you will obtain the updated version of the UI component information, including screenshots of the updated component (marked with green boxes indicating all updated UI components). You will need to analyze screenshots of two versions, to determine a best matched UI component's Number in Updated Screenshot.
# Finally, you need return the best matched UI component's Number in the Updated Screenshot.

# EXAMPLE OUTPUT:

# ```result.md
# ### Analyze_Process  
# Analyze the UI-related info between original and updated version of the UI component, and explain how to find the matched UI component's Number in Updated Screenshot.

# ### Matched_UI_No
# [18]
# ```

# """

#         user_prompt = {}

#         user_prompt['ori_analyze'] = [marked_original_img_base64] + [original_element_img_base64] + [
#             {"type": "text", "text": analyze_ori_scenarios_prompt}
#         ]

#         user_prompt['update_analyze'] = [marked_replay_img_base64] + [
#             {"type": "text", "text": analyze_update_effects_prompt}
#         ]

#         return system_prompt, user_prompt


# Jan 7, 81% success rate
#     def _construct_find_element_llm_prompt(self, marked_original_img_base64, original_element_img_base64, marked_replay_img_base64) -> Tuple[str, Dict]:
#         """
#         Construct LLM prompt
#         """

#         analyze_ori_scenarios_prompt = f"""
# I will provide you with the original application version's screenshot (marked with red boxes indicating an original UI component) and the original UI component Figure.


# * Original Screenshot
# ```
# please see the Figure 1.
# ```

# * Original UI component Figure
# ```
# please see the UI component figure.
# ```
# """
    
#         analyze_update_effects_prompt = f"""
# I will provide you with the updated application version's screenshot. Different UI components are marked with green boxes and assigned a numerical sequence number. You need to analyze the screenshots of the Updated version and identify an UI component that is most similar to the original UI component of the Original version, then provide the component sequence numbers.

# * Updated Screenshot
# ```
# please see the Figure 2.
# ```
# """
    
#         system_prompt = """
# You are an Android UI analysis expert. Your task is to identify which UI component in the UPDATED screen corresponds to the SAME FUNCTION as the original UI element.

# ## Core Matching Principle
# You must select the UI component that represents the SAME *functional meaning* as the original element — not necessarily the one that looks the most similar.

# Appearance, structure, and position may change across versions, but FUNCTION remains the anchor signal. Use visual/layout cues only as *secondary evidence*.

# ## What “Functional Match” Means
# When selecting the matched component, prioritize:

# 1. **Semantic equivalence**  
#    - Focus on the purpose of the control (e.g., enabling dark mode, opening a menu, choosing a theme, activating an option).
#    - The semantic meaning should be the closest to the original intent.

# 2. **Functional category alignment**  
#    UI components often evolve but stay in the same functional family.  
#    Examples:  
#    - checkbox → switch  
#    - switch → dialog containing radio choices  
#    - menu item → reorganized navigation entry  
#    - toggle → multi-option selector  

# 3. **Semantic strength alignment**  
#    When the original widget expresses a *strong meaning* (“Force Dark Theme”), and the updated version has multiple options:  
#    - pick the option that best matches the *assertive/off/on meaning*,  
#      **not** the one that merely preserves current selection.  
#    (Do NOT match based on which option is currently highlighted or selected.)

# 4. **State-independent matching**  
#    Ignore which option is currently selected in the updated version.  
#    Selection state is NOT part of functional equivalence.

# 5. **Auxiliary visual cues**  
#    Only use appearance, relative placement, or grouping to help disambiguate **after**
#    semantic meaning is considered.

# ## Important Constraints
# - Do NOT choose an option merely because it is currently selected.
# - Do NOT rely on visual similarity alone.
# - Do NOT rely on exact wording; wording may evolve (“Force Dark Theme” → “Dark”).
# - Your answer must reflect functional purpose, not UI form.

# ## Output
# Provide the NUMBER of the updated UI component that best matches the original element.

# Output format:
# ```result.md
# ### Analyze_Process
# (Your reasoning here)

# ### Matched_UI_No
# [18]
# ```

# """

#         user_prompt = {}

#         user_prompt['ori_analyze'] = [marked_original_img_base64] + [original_element_img_base64] + [
#             {"type": "text", "text": analyze_ori_scenarios_prompt}
#         ]

#         user_prompt['update_analyze'] = [marked_replay_img_base64] + [
#             {"type": "text", "text": analyze_update_effects_prompt}
#         ]

#         return system_prompt, user_prompt


    def _construct_find_element_llm_prompt(self, marked_original_img_base64, original_element_img_base64, marked_replay_img_base64, original_next_screen_summary=None) -> Tuple[str, Dict]:
        """
        Construct LLM prompt
        """

        analyze_ori_scenarios_prompt = f"""
I will provide you with the original version's screenshot (marked with red boxes indicating an original UI widget).

* Original Screenshot
```
Please see the above Figure.
```
"""
        analyze_ori_element_prompt = f"""
I will provide you with the original UI widget Figure.

* Original UI widget Figure
```
Please see the above Figure.
```
"""
    
        analyze_update_effects_prompt = f"""
I will provide you with the updated version's screenshot. Different UI widgets are marked with green boxes and assigned a numerical sequence number.

* Updated Screenshot
```
Please see the above Figure.
```
"""
    # 82.6% success rate
#         system_prompt = """
# You are an Android UI analysis expert with experience in UI evolution across app versions.
# Your task is to identify which UI widget on the updated screenshot corresponds to the original UI widget from the old version.

# ## Matching Principle
# - Select the UI widget that a user would most likely recognize as the same function, based on its purpose and role on the screen.
# - You may use the surrounding UI context on the current screen (e.g., page title, section grouping, nearby controls) to infer the role and intent of each UI widget.
# - Visual appearance, layout, or wording may change across versions. Use them only as supporting cues when intent is ambiguous.

# ## Output
# Provide the NUMBER of the updated UI widget that best matches the original widget.

# Output format:
# ```result.md
# ### Analyze_Process
# (Your reasoning here)

# ### Matched_UI_No
# [18]
# ```

# """

        system_prompt = """
You are an Android UI analysis expert with experience in UI evolution across app versions.
Your task is to identify which UI widget on the updated screenshot corresponds to the original UI widget from the old version.

## Matching Principle
- Select the UI widget that a user would most likely recognize as the same function, based on its purpose and role on the screen.
- You may use the surrounding UI context on the current screen (e.g., page title, section grouping, nearby controls) to infer the role and intent of each UI widget.
- Visual appearance, layout, or wording may change across versions. Use them only as supporting cues when intent is ambiguous.
- Do NOT consider any original checkbox/radio selection state. Only reason about the semantic purpose and function of the widget.
- Ensure the selected UI widget has the **same functional type** as the original:  
  - If the original is **editable** (e.g., checkbox, switch, radio, text field, option dialog), the matched widget must also be editable.  
  - If the original is **display-only** (e.g., label, static text, status indicator), the matched widget must also be display-only.
  - If the original is an **entry-type widget** (e.g., button, menu item, page entry, list item) that navigates to another screen or dialog, the matched widget must also be an entry-type widget.



## Output
Provide the NUMBER of the updated UI widget that best matches the original widget.

Output format:
```result.md
### Analyze_Process
(Your reasoning here)

### Matched_UI_No
[18]
```

"""

        user_prompt = {}

        user_prompt['ori_analyze'] = [marked_original_img_base64] + [{"type": "text", "text": analyze_ori_scenarios_prompt}] + [original_element_img_base64] + [{"type": "text", "text": analyze_ori_element_prompt}]

        if original_next_screen_summary:
            original_next_screen_block = f"""
## Original Next Screen
Below is the summary of the screen reached in the OLD version immediately AFTER interacting with the original UI element.

Use this screen to infer the semantic meaning of the ORIGINAL UI widget, especially when the original icon or label is abstract, by leveraging the resulting screen’s title and content.
{original_next_screen_summary}
            """
            
            user_prompt["ori_analyze"].append({"type": "text", "text": original_next_screen_block})


        user_prompt['update_analyze'] = [marked_replay_img_base64] + [
            {"type": "text", "text": analyze_update_effects_prompt}
        ]

        return system_prompt, user_prompt






import re
from typing import List, Dict
import uiautomator2
import xml.etree.ElementTree as ET
import rtree
from lxml import etree

class _HindenWidgetFilter:
    def __init__(self, root: etree._Element):
        # self.global_drawing_order = 0
        self._nodes = []

        self.idx = rtree.index.Index()
        try:
            self.set_covered_attr(root)
        except Exception as e:
            import traceback, uuid
            traceback.print_exc()

    def _iter_by_drawing_order(self, ele: etree._Element):
        """
        iter by drawing order (DFS)
        """
        if ele.tag == "node":
            yield ele

        children = list(ele)
        try:
            children.sort(key=lambda e: int(e.get("drawing-order", 0)))
        except (TypeError, ValueError):
            pass

        for child in children:
            yield from self._iter_by_drawing_order(child)

    def set_covered_attr(self, root: etree._Element):
        self._nodes: List[etree._Element] = list()
        for e in self._iter_by_drawing_order(root):
            # e.set("global-order", str(self.global_drawing_order))
            # self.global_drawing_order += 1
            e.set("covered", "false")

            # algorithm: filter by "clickable"
            clickable = (e.get("clickable", "false") == "true")
            _raw_bounds = e.get("bounds")
            if _raw_bounds is None:
                continue
            bounds = _get_bounds(_raw_bounds)
            if clickable:
                covered_widget_ids = list(self.idx.contains(bounds))
                if covered_widget_ids:
                    for covered_widget_id in covered_widget_ids:
                        node = self._nodes[covered_widget_id]
                        node.set("covered", "true")
                        self.idx.delete(
                            covered_widget_id,
                            _get_bounds(self._nodes[covered_widget_id].get("bounds"))
                        )

            cur_id = len(self._nodes)
            center = [
                (bounds[0] + bounds[2]) / 2,
                (bounds[1] + bounds[3]) / 2
            ]
            self.idx.insert(
                cur_id,
                (center[0], center[1], center[0], center[1])
            )
            self._nodes.append(e)

def _get_bounds(raw_bounds):
    pattern = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
    m = re.match(pattern, raw_bounds)
    try:
        bounds = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
    except Exception as e:
        print(f"raw_bounds: {raw_bounds}", flush=True)
        raise RuntimeError(e)

    return bounds

if __name__ == "__main__":
    llm_response = """
    ### Analyze_Process
The original UI component is a large, rounded rectangular blue button with a "+" (plus) sign. Its functional meaning is to increment or increase the counter displayed on the screen.

In the updated screenshot, the UI components are numbered 0 to 6. Components 2, 3, 4, 5, and 6 correspond to the top bar icons and title. Component 1 is a text prompt ("Try clicking on the number!"). Component 0 is the large number display showing "0" and is not interactive for incrementing.

There is no explicit "+" button visible in the updated screen. However, the text prompt (component 1) suggests that the user can increment the counter by clicking on the number itself (component 0). This indicates that the increment function that was originally a "+" button is now integrated into the number display UI element itself.

Thus, the component that functionally replaces the original "+" button is the number display area (component 0) which now also acts as the increment trigger.

### Matched_UI_No
[0]
    """
    element_id_str = get_component_no(llm_response)
    print(element_id_str)

    element_id = int(element_id_str)
    print(element_id)

    if element_id is not None and element_id>=0 and element_id<7:
        print("element_id is valid")
    else:
        print("element_id is invalid")
        
        
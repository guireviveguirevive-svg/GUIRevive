"""
AndroidTransitionAnalyzer - Python Implementation

Analyzes Android app transitions including:
- Explicit Intent transitions (Intent-based navigation to specific activities)
- Implicit Intent calls (system-level actions like share, view, etc.)
- More Options menu items
- Side bar (NavigationView) items

Usage:
    python transition_analyzer.py <source_code_path> <element_code_xml> <output_dir>

Output:
    {app_name}_Transition.xml
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement


def sanitize_xml_string(s: str) -> str:
    """Remove characters that are invalid in XML 1.0."""
    def is_valid_xml_char(c):
        code = ord(c)
        return (code == 0x9 or code == 0xA or code == 0xD or
                (0x20 <= code <= 0xD7FF) or
                (0xE000 <= code <= 0xFFFD) or
                (0x10000 <= code <= 0x10FFFF))
    return ''.join(c for c in s if is_valid_xml_char(c))


class StringAnalyzer:
    """Loads string resources from values/ directory."""

    def __init__(self, resource_path: str):
        self.string_map: Dict[str, str] = {}
        values_path = os.path.join(resource_path, "values")
        self._load_strings(values_path)

    def _load_strings(self, values_path: str):
        """Load all string resources from values directory."""
        if not os.path.exists(values_path):
            return

        for root, dirs, files in os.walk(values_path):
            for file in files:
                if file.endswith(".xml"):
                    self._load_string_file(os.path.join(root, file))

    def _load_string_file(self, file_path: str):
        """Load strings from a single XML file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for elem in root:
                if elem.tag == "string":
                    name = elem.get("name", "")
                    text = elem.text or ""
                    if name:
                        self.string_map[name] = text
        except ET.ParseError:
            pass

    def get_content(self, key: str) -> str:
        """Get string content by key."""
        return self.string_map.get(key, "")


class AndroidTransitionAnalyzer:
    """Analyzes Android app transitions and generates Transition.xml."""

    def __init__(self, element_code_path: str, app_name: str,
                 resource_path: str, code_path: str, pkg: str,
                 method_code_path: str):
        self.app_name = app_name
        self.resource_path = resource_path
        self.code_path = code_path
        self.pkg = pkg

        # Load ElementCode.xml
        self.element_code_root = self._load_xml(element_code_path)
        self.activity_list: List[str] = []  # Full component names
        self.activity_simple_names: List[str] = []  # Simple class names
        self.simple_to_component: Dict[str, str] = {}  # simple_name -> component_name
        self._get_activity_list()

        # Load MethodCode.xml
        self.method_code_root = self._load_xml(method_code_path)
        self.java_files: List[Element] = []
        if self.method_code_root is not None:
            self.java_files = list(self.method_code_root)

        # Load layout/menu/xml files
        self.layout_names: List[str] = []
        self.layout_paths: List[str] = []
        self.menu_names: List[str] = []
        self.menu_paths: List[str] = []
        self.xml_names: List[str] = []
        self.xml_paths: List[str] = []

        self._get_layout_files(os.path.join(resource_path, "layout"),
                               self.layout_names, self.layout_paths)
        self._get_layout_files(os.path.join(resource_path, "menu"),
                               self.menu_names, self.menu_paths)
        self._get_layout_files(os.path.join(resource_path, "xml"),
                               self.xml_names, self.xml_paths)

        # String analyzer
        self.string_analyzer = StringAnalyzer(resource_path)

        # Activity to layouts mapping
        self.activity_to_layouts: Dict[str, Set[str]] = {}
        self.fragment_set: Set[str] = set()

        # Menu item ID to element mapping
        self.id_to_menu_items: Dict[str, Element] = {}

        # Build mappings
        self._build_layout_list_by_activity()
        self._init_id_to_menu_items()

        # Output document
        self.transitions: List[Dict] = []

    def _load_xml(self, path: str) -> Optional[Element]:
        """Load and parse XML file."""
        if not os.path.exists(path):
            print(f"Warning: XML file not found: {path}")
            return None
        try:
            # Read file and sanitize
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            content = sanitize_xml_string(content)
            return ET.fromstring(content)
        except ET.ParseError as e:
            print(f"Error parsing XML {path}: {e}")
            return None

    def _get_activity_list(self):
        """Extract activity names from ElementCode.xml."""
        if self.element_code_root is None:
            return
        for activity in self.element_code_root:
            component_name = activity.get("name", "")
            if component_name:
                self.activity_list.append(component_name)
                # Extract simple class name from component name
                # e.g., "com.mirfatif.permissionmanagerx/.fwk.MainActivityM" -> "MainActivityM"
                simple_name = self._extract_simple_name(component_name)
                self.activity_simple_names.append(simple_name)
                self.simple_to_component[simple_name] = component_name

    def _extract_simple_name(self, component_name: str) -> str:
        """Extract simple class name from full component name."""
        # e.g., "com.mirfatif.permissionmanagerx/.fwk.MainActivityM" -> "MainActivityM"
        if '/' in component_name:
            component_name = component_name.split('/')[-1]
        if component_name.startswith('.'):
            component_name = component_name[1:]
        if '.' in component_name:
            component_name = component_name.split('.')[-1]
        return component_name

    def _get_component_name(self, simple_name: str) -> str:
        """Get full component name from simple class name."""
        return self.simple_to_component.get(simple_name, simple_name)

    def _get_layout_files(self, path: str, names: List[str], paths: List[str]):
        """Recursively find all XML files in a directory."""
        if not os.path.exists(path):
            return

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".xml"):
                    names.append(file.replace(".xml", ""))
                    paths.append(os.path.join(root, file))

    def _build_layout_list_by_activity(self):
        """Build mapping from activities/fragments to their layouts."""
        java_file_to_layouts: Dict[str, Set[str]] = {}

        for java_file in self.java_files:
            java_file_name = java_file.get("name", "").replace(".java", "").replace(".kt", "")
            layout_list: Set[str] = set()

            for method in java_file:
                method_content = method.text or ""

                # Check for layout/menu/xml references
                for layout_name in self.layout_names:
                    layout_id = f"R.layout.{layout_name}"
                    if layout_id in method_content:
                        layout_list.add(layout_name)

                for menu_name in self.menu_names:
                    menu_id = f"R.menu.{menu_name}"
                    if menu_id in method_content:
                        layout_list.add(menu_name)

                for xml_name in self.xml_names:
                    xml_id = f"R.xml.{xml_name}"
                    if xml_id in method_content:
                        layout_list.add(xml_name)

            # Add included layouts
            included_layouts = self._add_include_layouts(layout_list)
            layout_list.update(included_layouts)

            if layout_list or java_file_name in self.activity_simple_names:
                java_file_to_layouts[java_file_name] = layout_list

        # Identify fragments (java files with layouts but not in activity list)
        filtered_java_files = set(java_file_to_layouts.keys())
        fragments = {f for f in filtered_java_files if f not in self.activity_simple_names}
        self.fragment_set = fragments

        # Merge fragment layouts into activities that reference them
        for java_file in self.java_files:
            java_file_name = java_file.get("name", "").replace(".java", "").replace(".kt", "")
            if java_file_name not in filtered_java_files:
                continue

            for method in java_file:
                method_content = method.text or ""
                for fragment in fragments:
                    if fragment in method_content:
                        if java_file_name in java_file_to_layouts and fragment in java_file_to_layouts:
                            java_file_to_layouts[java_file_name].update(
                                java_file_to_layouts[fragment]
                            )

        self.activity_to_layouts = java_file_to_layouts

    def _add_include_layouts(self, layout_list: Set[str]) -> Set[str]:
        """Find all layouts included via <include> tags."""
        included_layouts: Set[str] = set()

        for layout_name in layout_list:
            if layout_name in self.layout_names:
                index = self.layout_names.index(layout_name)
                layout_path = self.layout_paths[index]

                tree_root = self._load_xml(layout_path)
                if tree_root is not None:
                    self._get_included_layouts(tree_root, included_layouts)

        return included_layouts

    def _get_included_layouts(self, node: Element, layout_list: Set[str]):
        """Recursively find included layouts."""
        if node.tag == "include":
            layout_ref = node.get("layout", "")
            if "/" in layout_ref:
                layout_name = layout_ref.split("/")[-1]
                layout_list.add(layout_name)

        for child in node:
            self._get_included_layouts(child, layout_list)

    def _init_id_to_menu_items(self):
        """Build mapping from menu item IDs to their elements."""
        for menu_path in self.menu_paths:
            tree_root = self._load_xml(menu_path)
            if tree_root is None:
                continue

            item_list: List[Element] = []
            self._get_menu_item_list(tree_root, item_list)

            for item in item_list:
                item_id = self._get_id(item)
                if item_id:
                    # Store with just the ID part (after /)
                    id_parts = item_id.split("/")
                    if len(id_parts) > 1:
                        self.id_to_menu_items[id_parts[-1]] = item

    def _get_menu_item_list(self, node: Element, item_list: List[Element]):
        """Recursively find all menu items."""
        # Handle both standard and namespaced item tags
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        if tag == "item":
            item_list.append(node)

        for child in node:
            self._get_menu_item_list(child, item_list)

    def _get_navigation_view_elements(self, node: Element, nav_list: List[Element]):
        """Recursively find NavigationView elements."""
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        if "NavigationView" in tag:
            nav_list.append(node)

        for child in node:
            self._get_navigation_view_elements(child, nav_list)

    def _get_id(self, element: Element) -> str:
        """Extract resource ID from element."""
        # Try both standard and android: namespace
        id_val = element.get("id") or element.get("{http://schemas.android.com/apk/res/android}id", "")
        if not id_val:
            return ""

        # Pattern: @+id/xxx or @id/xxx
        match = re.search(r"id/(.+)", id_val)
        if match:
            return f"{self.pkg}:id/{match.group(1)}"
        return ""

    def _get_title(self, element: Element) -> str:
        """Extract title resource name from element."""
        title = element.get("title") or element.get("{http://schemas.android.com/apk/res/android}title", "")
        if not title:
            return ""

        # Pattern: @string/xxx
        match = re.search(r"string/(.+)", title)
        if match:
            return match.group(1)
        return ""

    def _get_menu_attr(self, element: Element) -> str:
        """Extract menu attribute from NavigationView."""
        menu = element.get("menu") or element.get("{http://schemas.android.com/apk/res-auto}menu") or \
               element.get("{http://schemas.android.com/apk/res/android}menu", "")
        if menu and "/" in menu:
            return menu.split("/")[-1]
        return ""

    def analyze_activity_transition(self):
        """Analyze explicit Intent-based activity transitions."""
        if self.element_code_root is None:
            return

        for activity in self.element_code_root:
            source_activity = activity.get("name", "")  # This is already component name

            for ui_element in activity:
                for method in ui_element:
                    if method.tag != "method":
                        continue

                    method_content = method.get("body", "")

                    # Find all target activities from various Intent patterns
                    target_classes = self._extract_intent_targets(method_content)

                    for target_class in target_classes:
                        # Check if target is a known activity (using simple names)
                        if target_class in self.activity_simple_names:
                            target_name = self._get_component_name(target_class)
                            self._build_transition(source_activity, target_name, ui_element)

    def analyze_implicit_intents(self):
        """Analyze Implicit Intent calls (system-level actions) from all methods."""
        # Search through all methods in MethodCode.xml
        # We process all classes, not just activities, because utility classes
        # may also contain Intent calls
        for java_file in self.java_files:
            source_class = java_file.get("name", "").replace(".java", "").replace(".kt", "")

            for method in java_file:
                method_content = method.text or ""

                # Extract implicit intent information
                implicit_intents = self._extract_implicit_intents(method_content)

                for intent_info in implicit_intents:
                    # Try to map to an activity name, otherwise use class name
                    source_name = source_class
                    # Check if this class name matches any activity (using simple names)
                    for simple_name in self.activity_simple_names:
                        if simple_name in source_class or source_class in simple_name:
                            source_name = self._get_component_name(simple_name)
                            break

                    self._build_implicit_intent_transition_simple(
                        source_name, intent_info, method.get("name", "")
                    )

    def analyze_explicit_intents_from_all_methods(self):
        """
        Analyze Explicit Intent transitions by searching ALL methods.
        This complements analyze_activity_transition() which only searches
        UI-element-associated methods.
        """
        # Search through all methods in MethodCode.xml
        for java_file in self.java_files:
            source_class = java_file.get("name", "").replace(".java", "").replace(".kt", "")

            for method in java_file:
                method_content = method.text or ""

                # Extract explicit intent target activities
                target_classes = self._extract_intent_targets(method_content)

                for target_class in target_classes:
                    # Only record if target is a known activity (check simple names)
                    if target_class in self.activity_simple_names:
                        # Get component name for target
                        target_name = self._get_component_name(target_class)

                        # Try to map source to an activity name
                        source_name = source_class
                        for simple_name in self.activity_simple_names:
                            if simple_name in source_class or source_class in simple_name:
                                source_name = self._get_component_name(simple_name)
                                break

                        # Build transition with method info
                        self._build_transition_from_method(
                            source_name, target_name, method.get("name", "")
                        )

    def _extract_intent_targets(self, method_content: str) -> Set[str]:
        """
        Extract target Activity class names from various Intent patterns.
        Supports both Java and Kotlin styles.
        """
        targets = set()

        # Pattern 1: Java style - Intent(xxx, TargetActivity.class)
        java_pattern = re.compile(r'Intent\([^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)\.class\s*\)')
        for match in java_pattern.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 2: Kotlin style - Intent(xxx, TargetActivity::class.java)
        kotlin_intent_pattern = re.compile(r'Intent\([^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)::class\.java\s*\)')
        for match in kotlin_intent_pattern.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 3: Kotlin style - Intent(xxx, TargetActivity::class)
        kotlin_intent_pattern2 = re.compile(r'Intent\([^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)::class\s*\)')
        for match in kotlin_intent_pattern2.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 4: getIntent(context, TargetActivity::class, ...) - custom helper methods
        get_intent_pattern = re.compile(r'\.getIntent\([^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)::class')
        for match in get_intent_pattern.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 5: startActivity<TargetActivity>() - Kotlin inline
        start_activity_pattern = re.compile(r'startActivity\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>')
        for match in start_activity_pattern.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 6: intentFor<TargetActivity>() - Anko style
        intent_for_pattern = re.compile(r'intentFor\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>')
        for match in intent_for_pattern.finditer(method_content):
            targets.add(match.group(1))

        # Pattern 7: navigate to destination - could contain class reference
        # e.g., navigateTo(TargetActivity::class)
        navigate_pattern = re.compile(r'navigate\w*\([^)]*?([A-Za-z_][A-Za-z0-9_]*)::class')
        for match in navigate_pattern.finditer(method_content):
            targets.add(match.group(1))

        return targets

    def _extract_implicit_intents(self, method_content: str) -> List[Dict[str, str]]:
        """
        Extract Implicit Intent information from method code.
        Returns list of dicts with keys: action, data, type, extras
        """
        implicit_intents = []

        # Pattern 1: Intent(Intent.ACTION_XXX) single parameter
        action_constructor_single = re.compile(
            r'(?:new\s+)?Intent\s*\(\s*Intent\.([A-Z_]+)\s*\)'
        )

        # Pattern 2: Intent(Intent.ACTION_XXX, Uri.parse(...)) - with string literal
        action_constructor_with_data = re.compile(
            r'(?:new\s+)?Intent\s*\(\s*Intent\.([A-Z_]+)\s*,\s*Uri\.parse\s*\(\s*["\']([^"\']+)["\']\s*\)\s*\)'
        )

        # Pattern 2b: Intent(Intent.ACTION_XXX, Uri.parse(variable))
        action_constructor_with_uri_parse_var = re.compile(
            r'(?:new\s+)?Intent\s*\(\s*Intent\.([A-Z_]+)\s*,\s*Uri\.parse\s*\([^)]+\)\s*\)'
        )

        # Pattern 3: Intent(Intent.ACTION_XXX, uri_variable) - without Uri.parse
        action_constructor_with_uri_var = re.compile(
            r'(?:new\s+)?Intent\s*\(\s*Intent\.([A-Z_]+)\s*,\s*Uri\s*\)'
        )

        # Pattern 4: Intent("action.string")
        action_constructor_string = re.compile(
            r'(?:new\s+)?Intent\s*\(\s*["\']([a-z.]+[A-Z_]*[a-z.]*)["\']\s*\)'
        )

        # Pattern 5: intent.setAction(Intent.ACTION_XXX) or setAction("action.string")
        set_action_pattern = re.compile(
            r'\.setAction\s*\(\s*(?:Intent\.)?([A-Z_]+(?:\.[A-Z_]+)*|["\'][^"\']+["\'])\s*\)'
        )

        # Pattern 6: action = Intent.ACTION_XXX or action = "action.string"
        action_assignment_pattern = re.compile(
            r'action\s*=\s*Intent\.([A-Z_]+)'
        )

        # Pattern 7: Uri.parse("xxx")
        uri_pattern = re.compile(r'Uri\.parse\s*\(\s*["\']([^"\']+)["\']\s*\)')

        # Pattern 8: setType("type/subtype")
        type_pattern = re.compile(r'\.setType\s*\(\s*["\']([^"\']+)["\']\s*\)')

        # Pattern 9: type = "xxx"
        type_assignment_pattern = re.compile(r'type\s*=\s*["\']([^"\']+)["\']')

        # Pattern 10: setData(Uri.parse("xxx"))
        set_data_pattern = re.compile(r'\.setData\s*\(\s*Uri\.parse\s*\(\s*["\']([^"\']+)["\']\s*\)\s*\)')

        # Collect all matches
        actions = []
        uris = []
        types = []

        # Find actions from constructor - single parameter
        for match in action_constructor_single.finditer(method_content):
            actions.append(match.group(1))

        # Find actions from constructor - with URI data
        for match in action_constructor_with_data.finditer(method_content):
            actions.append(match.group(1))
            uris.append(match.group(2))

        # Find actions from constructor - with Uri.parse(variable)
        for match in action_constructor_with_uri_parse_var.finditer(method_content):
            actions.append(match.group(1))

        # Find actions from constructor - with URI variable (extract URI separately)
        for match in action_constructor_with_uri_var.finditer(method_content):
            action = match.group(1)
            # Only add if it doesn't look like a context variable
            if not action.startswith('this') and not action.startswith('context'):
                actions.append(action)

        # Find actions from string constructor
        for match in action_constructor_string.finditer(method_content):
            actions.append(match.group(1))

        # Find actions from setAction
        for match in set_action_pattern.finditer(method_content):
            action = match.group(1).strip('\'"')
            # Remove "Intent." prefix if present
            if '.' in action:
                actions.append(action.split('.')[-1])
            else:
                actions.append(action)

        # Find actions from assignment
        for match in action_assignment_pattern.finditer(method_content):
            actions.append(match.group(1))

        # Find URIs from Uri.parse()
        for match in uri_pattern.finditer(method_content):
            uris.append(match.group(1))

        # Find URIs from setData()
        for match in set_data_pattern.finditer(method_content):
            uris.append(match.group(1))

        # Find types from setType()
        for match in type_pattern.finditer(method_content):
            types.append(match.group(1))

        # Find types from assignment
        for match in type_assignment_pattern.finditer(method_content):
            types.append(match.group(1))

        # Create intent records - one per unique action
        seen_actions = set()
        for action in actions:
            if action and action not in seen_actions:
                seen_actions.add(action)

                intent_info = {'action': action}

                # Try to match URIs (use first available)
                if uris:
                    intent_info['data'] = uris[0]

                # Try to match types (use first available)
                if types:
                    intent_info['type'] = types[0]

                implicit_intents.append(intent_info)

        return implicit_intents

    def analyze_more_options(self):
        """Analyze toolbar/options menu items."""
        for java_file in self.java_files:
            source_simple = java_file.get("name", "").replace(".java", "").replace(".kt", "")

            if source_simple not in self.activity_simple_names and source_simple not in self.fragment_set:
                continue

            # Convert to component name for output
            source = self._get_component_name(source_simple)

            is_on_create_options_menu_found = False
            methods = list(java_file)

            for method in methods:
                method_name = method.get("name", "")
                method_content = method.text or ""

                # onCreateOptionsMenu inflates menu
                if method_name == "onCreateOptionsMenu":
                    is_on_create_options_menu_found = True
                    pattern = re.compile(r'R\.menu\.([a-zA-Z0-9_-]+)')
                    for match in pattern.finditer(method_content):
                        menu_name = match.group(1)
                        self._build_more_options_transition(source, menu_name)

                # toolbar.inflateMenu(R.menu.xxx)
                toolbar_pattern = re.compile(r'(?:toolbar|Toolbar)\.inflateMenu\s*\(\s*R\.menu\.([^)]+)\)')
                for match in toolbar_pattern.finditer(method_content):
                    menu_name = match.group(1).strip()
                    self._build_more_options_transition(source, menu_name)

            # If no onCreateOptionsMenu, check onOptionsItemSelected for menu IDs
            if not is_on_create_options_menu_found:
                for method in methods:
                    method_name = method.get("name", "")
                    method_content = method.text or ""

                    if method_name == "onOptionsItemSelected":
                        pattern = re.compile(r'R\.id\.([a-zA-Z0-9_-]+)')
                        menu_ids = [match.group(1) for match in pattern.finditer(method_content)]
                        if menu_ids:
                            self._build_more_options_transition_by_ids(source, menu_ids)

    def analyze_side_bar(self):
        """Analyze NavigationView side bar items."""
        # Iterate over simple names since activity_to_layouts uses simple names as keys
        for simple_name in self.activity_simple_names:
            layouts = self.activity_to_layouts.get(simple_name, set())

            # Get component name for output
            activity = self._get_component_name(simple_name)

            for layout in layouts:
                if layout not in self.layout_names:
                    continue

                layout_index = self.layout_names.index(layout)
                layout_path = self.layout_paths[layout_index]

                layout_root = self._load_xml(layout_path)
                if layout_root is None:
                    continue

                nav_view_list: List[Element] = []
                self._get_navigation_view_elements(layout_root, nav_view_list)

                for nav_view in nav_view_list:
                    menu = self._get_menu_attr(nav_view)
                    if menu:
                        self._build_side_bar_transition(activity, menu)

    def _build_transition(self, source: str, target: str, ui_element: Element):
        """Build an activity transition record."""
        transition = {
            "type": "Path",
            "source": source,
            "target": target,
            "element": {}
        }

        tag = ui_element.tag
        if tag == "UIElement":
            transition["element"] = {
                "type": "UIElement",
                "id": ui_element.get("id", ""),
                "text": ui_element.get("text", ""),
                "textContent": ui_element.get("textContent", "")
            }
        elif tag == "Preference":
            transition["element"] = {
                "type": "Preference",
                "key": ui_element.get("key", ""),
                "title": ui_element.get("title", ""),
                "titleContent": ui_element.get("titleContent", "")
            }

        self.transitions.append(transition)

    def _build_implicit_intent_transition(self, source: str, intent_info: Dict[str, str],
                                          ui_element: Element):
        """Build an implicit intent transition record (with UI element)."""
        transition = {
            "type": "ImplicitIntent",
            "source": source,
            "intent": {
                "action": intent_info.get("action", ""),
                "data": intent_info.get("data", ""),
                "type": intent_info.get("type", "")
            },
            "element": {}
        }

        tag = ui_element.tag
        if tag == "UIElement":
            transition["element"] = {
                "type": "UIElement",
                "id": ui_element.get("id", ""),
                "text": ui_element.get("text", ""),
                "textContent": ui_element.get("textContent", "")
            }
        elif tag == "Preference":
            transition["element"] = {
                "type": "Preference",
                "key": ui_element.get("key", ""),
                "title": ui_element.get("title", ""),
                "titleContent": ui_element.get("titleContent", "")
            }

        self.transitions.append(transition)

    def _build_implicit_intent_transition_simple(self, source: str, intent_info: Dict[str, str],
                                                 method_name: str):
        """Build an implicit intent transition record (without UI element)."""
        transition = {
            "type": "ImplicitIntent",
            "source": source,
            "method": method_name,
            "intent": {
                "action": intent_info.get("action", ""),
                "data": intent_info.get("data", ""),
                "type": intent_info.get("type", "")
            }
        }

        self.transitions.append(transition)

    def _build_transition_from_method(self, source: str, target: str, method_name: str):
        """Build an explicit intent transition record from method (without UI element)."""
        transition = {
            "type": "Path",
            "source": source,
            "target": target,
            "method": method_name,
            "element": {}  # No UI element since we found this in a non-UI method
        }

        self.transitions.append(transition)

    def _build_more_options_transition(self, source: str, menu_name: str):
        """Build a more options transition from menu file."""
        if menu_name not in self.menu_names:
            return

        menu_index = self.menu_names.index(menu_name)
        menu_path = self.menu_paths[menu_index]

        menu_root = self._load_xml(menu_path)
        if menu_root is None:
            return

        item_list: List[Element] = []
        self._get_menu_item_list(menu_root, item_list)

        items = []
        for item in item_list:
            item_id = self._get_id(item)
            title = self._get_title(item)
            title_content = self.string_analyzer.get_content(title)

            items.append({
                "id": item_id,
                "title": title,
                "titleContent": title_content
            })

        if items:
            self.transitions.append({
                "type": "MoreOptions",
                "source": source,
                "items": items
            })

    def _build_more_options_transition_by_ids(self, source: str, menu_ids: List[str]):
        """Build a more options transition from menu item IDs."""
        items = []

        for menu_id in menu_ids:
            item_element = self.id_to_menu_items.get(menu_id)
            if item_element is None:
                continue

            item_id = f"{self.pkg}:id/{menu_id}"
            title = self._get_title(item_element)
            title_content = self.string_analyzer.get_content(title)

            items.append({
                "id": item_id,
                "title": title,
                "titleContent": title_content
            })

        if items:
            self.transitions.append({
                "type": "MoreOptions",
                "source": source,
                "items": items
            })

    def _build_side_bar_transition(self, source: str, menu_name: str):
        """Build a sidebar transition from menu file."""
        if menu_name not in self.menu_names:
            return

        menu_index = self.menu_names.index(menu_name)
        menu_path = self.menu_paths[menu_index]

        menu_root = self._load_xml(menu_path)
        if menu_root is None:
            return

        item_list: List[Element] = []
        self._get_menu_item_list(menu_root, item_list)

        items = []
        for item in item_list:
            item_id = self._get_id(item)
            title = self._get_title(item)
            title_content = self.string_analyzer.get_content(title)

            items.append({
                "id": item_id,
                "title": title,
                "titleContent": title_content
            })

        if items:
            self.transitions.append({
                "type": "SideBar",
                "source": source,
                "items": items
            })

    def generate_xml(self, output_path: str):
        """Generate the Transition.xml output file."""
        root = Element("TransitionGraph")

        for transition in self.transitions:
            trans_type = transition["type"]

            if trans_type == "ImplicitIntent":
                intent_elem = SubElement(root, "ImplicitIntent")

                source_elem = SubElement(intent_elem, "source")
                source_elem.set("name", transition["source"])

                intent_info = transition.get("intent", {})
                intent_details = SubElement(intent_elem, "intent")

                if intent_info.get("action"):
                    intent_details.set("action", intent_info["action"])
                if intent_info.get("data"):
                    intent_details.set("data", intent_info["data"])
                if intent_info.get("type"):
                    intent_details.set("type", intent_info["type"])

                # Add method name if available
                if transition.get("method"):
                    method_elem = SubElement(intent_elem, "method")
                    method_elem.set("name", transition["method"])

                # Add UI element if available
                elem_info = transition.get("element", {})
                if elem_info:
                    elem_type = elem_info.get("type", "UIElement")
                    ui_elem = SubElement(intent_elem, elem_type)

                    if elem_type == "UIElement":
                        ui_elem.set("id", elem_info.get("id", ""))
                        ui_elem.set("text", elem_info.get("text", ""))
                        ui_elem.set("textContent", elem_info.get("textContent", ""))
                    else:  # Preference
                        ui_elem.set("key", elem_info.get("key", ""))
                        ui_elem.set("title", elem_info.get("title", ""))
                        ui_elem.set("titleContent", elem_info.get("titleContent", ""))

            elif trans_type == "Path":
                path_elem = SubElement(root, "Path")

                source_elem = SubElement(path_elem, "source")
                source_elem.set("name", transition["source"])

                target_elem = SubElement(path_elem, "target")
                target_elem.set("name", transition["target"])

                elem_info = transition.get("element", {})
                if elem_info:
                    elem_type = elem_info.get("type", "UIElement")
                    ui_elem = SubElement(path_elem, elem_type)

                    if elem_type == "UIElement":
                        ui_elem.set("id", elem_info.get("id", ""))
                        ui_elem.set("text", elem_info.get("text", ""))
                        ui_elem.set("textContent", elem_info.get("textContent", ""))
                    else:  # Preference
                        ui_elem.set("key", elem_info.get("key", ""))
                        ui_elem.set("title", elem_info.get("title", ""))
                        ui_elem.set("titleContent", elem_info.get("titleContent", ""))

                # Add method name if this transition was found from a method (not UI element)
                method_name = transition.get("method", "")
                if method_name:
                    method_elem = SubElement(path_elem, "method")
                    method_elem.set("name", method_name)

            elif trans_type == "MoreOptions":
                more_opts_elem = SubElement(root, "MoreOptions")

                source_elem = SubElement(more_opts_elem, "source")
                source_elem.set("name", transition["source"])

                for item in transition.get("items", []):
                    item_elem = SubElement(more_opts_elem, "item")
                    item_elem.set("id", item.get("id", ""))
                    item_elem.set("title", item.get("title", ""))
                    item_elem.set("titleContent", item.get("titleContent", ""))

            elif trans_type == "SideBar":
                sidebar_elem = SubElement(root, "SideBar")

                source_elem = SubElement(sidebar_elem, "source")
                source_elem.set("name", transition["source"])

                for item in transition.get("items", []):
                    item_elem = SubElement(sidebar_elem, "item")
                    item_elem.set("id", item.get("id", ""))
                    item_elem.set("title", item.get("title", ""))
                    item_elem.set("titleContent", item.get("titleContent", ""))

        # Write to file with formatting
        self._write_xml(root, output_path)
        print(f"Output saved to: {output_path}")

    def _write_xml(self, root: Element, output_path: str):
        """Write XML with proper formatting."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            self._write_element(f, root, 0)

    def _write_element(self, f, elem: Element, level: int):
        """Recursively write XML element with indentation."""
        indent = "  " * level

        # Start tag with attributes
        attrs = ' '.join(f'{k}="{v}"' for k, v in elem.attrib.items())
        if attrs:
            f.write(f"{indent}<{elem.tag} {attrs}")
        else:
            f.write(f"{indent}<{elem.tag}")

        if len(elem) == 0 and not elem.text:
            f.write("/>\n")
            return

        f.write(">")

        if elem.text and elem.text.strip():
            f.write(sanitize_xml_string(elem.text))

        if len(elem) > 0:
            f.write("\n")
            for child in elem:
                self._write_element(f, child, level + 1)
            f.write(indent)

        f.write(f"</{elem.tag}>\n")

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about analyzed transitions."""
        stats = {
            "activities": len(self.activity_list),
            "fragments": len(self.fragment_set),
            "layouts": len(self.layout_names),
            "menus": len(self.menu_names),
            "path_transitions": 0,
            "implicit_intents": 0,
            "more_options": 0,
            "side_bars": 0,
            "total_menu_items": 0
        }

        for trans in self.transitions:
            if trans["type"] == "Path":
                stats["path_transitions"] += 1
            elif trans["type"] == "ImplicitIntent":
                stats["implicit_intents"] += 1
            elif trans["type"] == "MoreOptions":
                stats["more_options"] += 1
                stats["total_menu_items"] += len(trans.get("items", []))
            elif trans["type"] == "SideBar":
                stats["side_bars"] += 1
                stats["total_menu_items"] += len(trans.get("items", []))

        return stats


def find_android_paths(source_code_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-discover resource path and package name from source code path.
    Returns (resource_path, package_name)

    Prefers the module with the most layout files (typically the main app module).
    """
    resource_path = None
    pkg = None

    # Find all potential resource paths (directories with res/layout)
    candidates = []
    for root, dirs, files in os.walk(source_code_path):
        if "AndroidManifest.xml" in files:
            res_path = os.path.join(root, "res")
            layout_path = os.path.join(res_path, "layout")
            if os.path.isdir(layout_path):
                # Count layout files to find main app module
                layout_count = len([f for f in os.listdir(layout_path) if f.endswith(".xml")])
                candidates.append((res_path, root, layout_count))

    # Choose the candidate with most layout files
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        resource_path = candidates[0][0]
        manifest_dir = candidates[0][1]

        # Try to get package from manifest
        manifest_path = os.path.join(manifest_dir, "AndroidManifest.xml")
        try:
            tree = ET.parse(manifest_path)
            manifest_root = tree.getroot()
            pkg = manifest_root.get("package", "")
        except:
            pass

    # If no package found, try build.gradle.kts in the module directory
    if not pkg and resource_path:
        module_dir = os.path.dirname(os.path.dirname(os.path.dirname(resource_path)))
        for gradle_file in ["build.gradle.kts", "build.gradle"]:
            gradle_path = os.path.join(module_dir, gradle_file)
            if os.path.exists(gradle_path):
                try:
                    with open(gradle_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for variable definitions first (like APP_ID = '...')
                    var_matches = re.findall(r'(?:String|def|val|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']([^"\']+)["\']', content)
                    var_dict = {name: value for name, value in var_matches}
                    
                    # Look for namespace with variable
                    ns_var_match = re.search(r'namespace\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                    if ns_var_match and ns_var_match.group(1) in var_dict:
                        pkg = var_dict[ns_var_match.group(1)]
                        break
                    
                    # Look for applicationId with variable
                    app_id_var_match = re.search(r'applicationId\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                    if app_id_var_match and app_id_var_match.group(1) in var_dict:
                        pkg = var_dict[app_id_var_match.group(1)]
                        break
                    
                    # Standard patterns
                    match = re.search(r'namespace\s*=?\s*["\']([^"\']+)["\']', content)
                    if match:
                        pkg = match.group(1)
                        break
                    match = re.search(r'applicationId\s*=?\s*["\']([^"\']+)["\']', content)
                    if match:
                        pkg = match.group(1)
                        break
                    
                    # Look for package name in plugins block (Kotlin DSL)
                    match = re.search(r'id\s*\(\s*["\']com\.android\.[^"\']+["\']\s*\)[\s\S]*?namespace\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        pkg = match.group(1)
                        break
                except Exception as e:
                    print(f"  Error parsing gradle file {gradle_path}: {e}")

    # Fallback: search all gradle files
    if not pkg:
        for root, dirs, files in os.walk(source_code_path):
            for file in files:
                if file in ["build.gradle.kts", "build.gradle"]:
                    gradle_path = os.path.join(root, file)
                    try:
                        with open(gradle_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Look for variable definitions first (like APP_ID = '...')
                        var_matches = re.findall(r'(?:String|def|val|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']([^"\']+)["\']', content)
                        var_dict = {name: value for name, value in var_matches}
                        
                        # Look for namespace with variable
                        ns_var_match = re.search(r'namespace\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                        if ns_var_match and ns_var_match.group(1) in var_dict:
                            pkg = var_dict[ns_var_match.group(1)]
                            break
                        
                        # Look for applicationId with variable
                        app_id_var_match = re.search(r'applicationId\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                        if app_id_var_match and app_id_var_match.group(1) in var_dict:
                            pkg = var_dict[app_id_var_match.group(1)]
                            break
                        
                        # Standard patterns
                        match = re.search(r'namespace\s*=?\s*["\']([^"\']+)["\']', content)
                        if match:
                            pkg = match.group(1)
                            break
                        match = re.search(r'applicationId\s*=?\s*["\']([^"\']+)["\']', content)
                        if match:
                            pkg = match.group(1)
                            break
                    except Exception as e:
                        print(f"  Error parsing gradle file {gradle_path}: {e}")
            if pkg:
                break
                
    # If still no package found, use a hardcoded default for known projects
    if not pkg:
        if "PermissionManagerX" in source_code_path:
            pkg = "com.mirfatif.permissionmanagerx"
            print(f"  Using hardcoded package name for PermissionManagerX: {pkg}")
        elif "Anki-Android" in source_code_path:
            pkg = "com.ichi2.anki"
            print(f"  Using hardcoded package name for Anki-Android: {pkg}")

    return resource_path, pkg


def main():
    if len(sys.argv) < 3:
        print("Usage: python transition_analyzer.py <source_code_path> <element_code_xml> [output_dir]")
        print("Example: python transition_analyzer.py ./Anki-Android-2.23.1 ./output/Anki-Android-2.23.1_ElementCode.xml ./output")
        sys.exit(1)

    source_code_path = sys.argv[1].rstrip('/')
    element_code_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "."

    if not os.path.exists(source_code_path):
        print(f"Error: Source path '{source_code_path}' does not exist")
        sys.exit(1)

    if not os.path.exists(element_code_path):
        print(f"Error: ElementCode XML '{element_code_path}' does not exist")
        sys.exit(1)

    # Extract app name from element code filename
    app_name = os.path.basename(element_code_path).replace("_ElementCode.xml", "")

    # Auto-discover paths
    resource_path, pkg = find_android_paths(source_code_path)

    if not resource_path:
        print("Error: Could not find Android resource path")
        sys.exit(1)

    if not pkg:
        pkg = "unknown.package"
        print(f"Warning: Could not determine package name, using '{pkg}'")

    # Method code path (same directory as element code)
    element_code_dir = os.path.dirname(element_code_path)
    method_code_path = os.path.join(element_code_dir, f"{app_name}_MethodCode.xml")

    if not os.path.exists(method_code_path):
        print(f"Error: MethodCode XML '{method_code_path}' does not exist")
        sys.exit(1)

    print(f"Source code path: {source_code_path}")
    print(f"Resource path: {resource_path}")
    print(f"Package: {pkg}")
    print(f"ElementCode: {element_code_path}")
    print(f"MethodCode: {method_code_path}")
    print("-" * 50)

    # Create analyzer
    analyzer = AndroidTransitionAnalyzer(
        element_code_path=element_code_path,
        app_name=app_name,
        resource_path=resource_path,
        code_path=source_code_path,
        pkg=pkg,
        method_code_path=method_code_path
    )

    # Analyze transitions
    print("Analyzing explicit intent transitions (UI-based)...")
    analyzer.analyze_activity_transition()

    print("Analyzing explicit intent transitions (all methods)...")
    analyzer.analyze_explicit_intents_from_all_methods()

    print("Analyzing implicit intent calls...")
    analyzer.analyze_implicit_intents()

    print("Analyzing more options menus...")
    analyzer.analyze_more_options()

    print("Analyzing side bar navigation...")
    analyzer.analyze_side_bar()

    # Generate output
    output_path = os.path.join(output_dir, f"{app_name}_Transition.xml")
    analyzer.generate_xml(output_path)

    # Print statistics
    stats = analyzer.get_statistics()
    print("-" * 50)
    print("Statistics:")
    print(f"  Activities: {stats['activities']}")
    print(f"  Fragments: {stats['fragments']}")
    print(f"  Layouts: {stats['layouts']}")
    print(f"  Menus: {stats['menus']}")
    print(f"  Explicit intent transitions: {stats['path_transitions']}")
    print(f"  Implicit intent calls: {stats['implicit_intents']}")
    print(f"  MoreOptions menus: {stats['more_options']}")
    print(f"  SideBar menus: {stats['side_bars']}")
    print(f"  Total menu items: {stats['total_menu_items']}")


if __name__ == '__main__':
    main()

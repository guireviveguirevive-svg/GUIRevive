"""
AndroidMethodCodeExtractor - Python Implementation

Extracts method names and bodies from Java and Kotlin source files.
Equivalent to the original AndroidMethodCodeExtractor.jar (extended for Kotlin)

Usage:
    python method_code_extractor.py <source_code_path> [output_dir]

Output:
    {app_name}_MethodCode.xml (app_name is extracted from source_code_path)
"""

import os
import re
import sys
import javalang
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
from typing import List, Tuple


def sanitize_xml_string(s: str) -> str:
    """
    Remove characters that are invalid in XML 1.0.
    Valid XML 1.0 characters: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD]
    """
    def is_valid_xml_char(c):
        code = ord(c)
        return (code == 0x9 or code == 0xA or code == 0xD or
                (0x20 <= code <= 0xD7FF) or
                (0xE000 <= code <= 0xFFFD) or
                (0x10000 <= code <= 0x10FFFF))

    return ''.join(c for c in s if is_valid_xml_char(c))


def get_source_files(root_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Recursively find all .java and .kt files in the given directory.

    Args:
        root_path: Root directory to search

    Returns:
        Tuple of (java_files, kotlin_files), each is a list of (filename, filepath)
    """
    java_files = []
    kotlin_files = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename.endswith('.java'):
                java_files.append((filename, filepath))
            elif filename.endswith('.kt'):
                kotlin_files.append((filename, filepath))

    return java_files, kotlin_files


def extract_methods_java(file_path: str) -> List[Tuple[str, str]]:
    """
    Extract method names and bodies from a Java file.

    Args:
        file_path: Path to the Java file

    Returns:
        List of tuples (method_name, method_body)
    """
    methods = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()

        # Parse the Java source code
        tree = javalang.parse.parse(source_code)

        # Find all method declarations
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name
            method_body = extract_method_body_java(source_code, node)

            if method_body:
                methods.append((method_name, method_body))

    except javalang.parser.JavaSyntaxError as e:
        print(f"  Syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

    return methods


def extract_method_body_java(source_code: str, method_node) -> str:
    """
    Extract the method body from Java source code based on the method node position.
    """
    if method_node.position is None:
        return ""

    lines = source_code.split('\n')
    start_line = method_node.position.line - 1  # 0-indexed

    # Find the opening brace of the method body
    brace_count = 0
    started = False
    body_lines = []

    for i in range(start_line, len(lines)):
        line = lines[i]

        for char in line:
            if char == '{':
                brace_count += 1
                started = True
            elif char == '}':
                brace_count -= 1

        if started:
            body_lines.append(line)

        if started and brace_count == 0:
            break

    if not body_lines:
        return ""

    body = '\n'.join(body_lines)

    # Remove the method signature part, keep only the body
    first_brace = body.find('{')
    if first_brace != -1:
        body = body[first_brace + 1:]
        last_brace = body.rfind('}')
        if last_brace != -1:
            body = body[:last_brace]

    return body.strip()


def extract_methods_kotlin(file_path: str) -> List[Tuple[str, str]]:
    """
    Extract function names and bodies from a Kotlin file.

    Args:
        file_path: Path to the Kotlin file

    Returns:
        List of tuples (function_name, function_body)
    """
    methods = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()

        # Find all function declarations using regex
        # Matches: fun functionName(...) { ... } or fun functionName(...) = ...
        # Also matches: private fun, override fun, suspend fun, etc.

        # Pattern for function with body in braces
        fun_pattern = re.compile(
            r'(?:(?:private|public|internal|protected|override|suspend|inline|open|abstract|final)\s+)*'
            r'fun\s+(?:<[^>]+>\s*)?'  # generic type parameters
            r'(\w+)\s*'  # function name (captured)
            r'\([^)]*\)'  # parameters
            r'(?:\s*:\s*[^\{=]+)?'  # optional return type
            r'\s*(\{|=)',  # body start: { or =
            re.MULTILINE
        )

        for match in fun_pattern.finditer(source_code):
            func_name = match.group(1)
            body_start_char = match.group(2)
            start_pos = match.end()

            if body_start_char == '{':
                # Extract body within braces
                body = extract_braced_body(source_code, start_pos - 1)
            else:
                # Expression body (= ...)
                body = extract_expression_body(source_code, start_pos)

            if body:
                methods.append((func_name, body.strip()))

    except Exception as e:
        print(f"  Error processing Kotlin file {file_path}: {e}")

    return methods


def extract_braced_body(source_code: str, start_pos: int) -> str:
    """
    Extract body content within braces starting from start_pos.
    """
    brace_count = 0
    started = False
    body_start = start_pos
    body_end = start_pos

    for i in range(start_pos, len(source_code)):
        char = source_code[i]

        if char == '{':
            if not started:
                body_start = i + 1
                started = True
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if started and brace_count == 0:
            body_end = i
            break

    if not started:
        return ""

    return source_code[body_start:body_end].strip()


def extract_expression_body(source_code: str, start_pos: int) -> str:
    """
    Extract expression body (after =) until end of statement.
    Handles multi-line expressions and nested structures.
    """
    # Find the end of the expression
    # Expression ends at newline (unless inside braces/parens/brackets)
    paren_count = 0
    brace_count = 0
    bracket_count = 0
    body_chars = []
    in_string = False
    string_char = None

    for i in range(start_pos, len(source_code)):
        char = source_code[i]
        prev_char = source_code[i - 1] if i > 0 else ''

        # Handle string literals
        if char in '"\'`' and prev_char != '\\':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False

        if not in_string:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == '\n' and paren_count == 0 and brace_count == 0 and bracket_count == 0:
                # Check if next non-whitespace continues the expression
                rest = source_code[i + 1:].lstrip()
                if not rest or not rest[0] in '.?:+-*/%&|':
                    break

        body_chars.append(char)

    return ''.join(body_chars).strip()


def generate_method_code_xml(source_files: List[Tuple[str, str, str]],
                              output_path: str) -> None:
    """
    Generate MethodCode.xml from extracted methods.

    Args:
        source_files: List of (filename, filepath, filetype) tuples
        output_path: Output XML file path
    """
    root = Element('MethodCode')

    if not source_files:
        print("Warning: No source files found!")

    for filename, filepath, filetype in source_files:
        if filetype == 'java':
            methods = extract_methods_java(filepath)
        else:  # kotlin
            methods = extract_methods_kotlin(filepath)

        if not methods:
            continue

        # Use 'SourceFile' as element name to be more generic
        # But keep 'JavaFile' for compatibility with original format
        file_elem = SubElement(root, 'JavaFile')
        file_elem.set('name', filename)

        for method_name, method_body in methods:
            method_elem = SubElement(file_elem, 'Method')
            method_elem.set('name', method_name)
            method_elem.text = method_body

    # Write with proper formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        write_element(f, root, 0)

    print(f"\nOutput saved to: {output_path}")


def write_element(f, elem: Element, level: int) -> None:
    """
    Recursively write XML element with proper indentation.
    """
    indent = "  " * level

    # Start tag
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
        # For method body, use CDATA with sanitized content
        if elem.tag == 'Method':
            sanitized_text = sanitize_xml_string(elem.text)
            f.write(f"<![CDATA[{sanitized_text}]]>")
        else:
            f.write(sanitize_xml_string(elem.text))

    if len(elem) > 0:
        f.write("\n")
        for child in elem:
            write_element(f, child, level + 1)
        f.write(indent)

    f.write(f"</{elem.tag}>\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python method_code_extractor.py <source_code_path> [output_dir]")
        print("Example: python method_code_extractor.py ./Anki-Android-2.23.1")
        print("\nSupports both Java (.java) and Kotlin (.kt) files.")
        print("The app name is automatically extracted from the source path.")
        sys.exit(1)

    source_path = sys.argv[1].rstrip('/')
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(source_path):
        print(f"Error: Path '{source_path}' does not exist")
        sys.exit(1)

    # Extract app name from path (last directory name)
    app_name = os.path.basename(source_path)

    # If output_dir is None, use the parent directory of source_path
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(source_path))
        if not output_dir:  # If source_path is in current dir
            output_dir = '.'

    print(f"Extracting methods from: {source_path}")
    print(f"App name: {app_name}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Find all source files
    java_files, kotlin_files = get_source_files(source_path)
    print(f"Found {len(java_files)} Java files, {len(kotlin_files)} Kotlin files")

    # Combine into single list with file type marker
    source_files = []
    for filename, filepath in java_files:
        source_files.append((filename, filepath, 'java'))
    for filename, filepath in kotlin_files:
        source_files.append((filename, filepath, 'kotlin'))

    # Generate output XML
    output_path = os.path.join(output_dir, f"{app_name}_MethodCode.xml")
    generate_method_code_xml(source_files, output_path)


if __name__ == '__main__':
    main()

import xml.etree.ElementTree as ET
from typing import Tuple, Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont, ImageOps
import difflib
import math
import base64
from math import sqrt
from openai import OpenAI
import re
import os
from bs4 import BeautifulSoup


def trim_whitespace(image, threshold=235):
    """
    Trim whitespace around the image

    Args:
        image: PIL Image object
        threshold: Pixel brightness threshold; values above this are considered whitespace

    Returns:
        The cropped image
    """
    # Convert to grayscale
    img_gray = image.convert('L')
    img_array = np.array(img_gray)
    
    # Find non-whitespace regions
    mask = img_array < threshold

    # If the entire image is whitespace, return the original image
    if not np.any(mask):
        return image
    
    # Find the boundaries of non-whitespace regions
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Ensure the crop area has at least a 1-pixel margin
    y_min = max(0, y_min - 1)
    y_max = min(img_array.shape[0] - 1, y_max + 1)
    x_min = max(0, x_min - 1)
    x_max = min(img_array.shape[1] - 1, x_max + 1)
    
    # Crop the image
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))


def compute_ssim(image1, image2, trim=True):
    """
    Compute the structural similarity of two images

    Args:
        image1: First PIL Image object
        image2: Second PIL Image object
        trim: Whether to trim whitespace, defaults to True

    Returns:
        Structural similarity score (0-1)
    """
    # Trim whitespace
    # if trim:
    #     image1 = trim_whitespace(image1)
    #     image2 = trim_whitespace(image2)

    # Add brightness equalization
    image1 = ImageOps.equalize(image1.convert('L'))
    image2 = ImageOps.equalize(image2.convert('L'))
    
    # Convert to grayscale and resize
    image1 = image1.convert('L').resize((128, 128))
    image2 = image2.convert('L').resize((128, 128))
    
    # Convert to NumPy arrays
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    
    # Compute SSIM
    score, _ = ssim(arr1, arr2, full=True)
    return round(score, 4)

def read_image(image_path):
    image = Image.open(image_path)
    return image

def draw_element_on_image(image, bounds):
    """
    Draw element bounds on the image

    Args:
        image: PIL Image object or image path
        bounds: Element bounds, in format (x1, y1, x2, y2) or "[x1,y1][x2,y2]"
        id: Element ID number, defaults to None
    Returns:
        The annotated image object
    """
    image = image.copy()  # Prevent modifying the original image
    # Create drawing object
    draw = ImageDraw.Draw(image)
    
    # Parse bounds
    if isinstance(bounds, str):
        # If in string format "[x1,y1][x2,y2]", parse into coordinates
        import re
        match = re.findall(r"\[(\d+),(\d+)\]", bounds)
        if match and len(match) == 2:
            x1, y1 = int(match[0][0]), int(match[0][1])
            x2, y2 = int(match[1][0]), int(match[1][1])
            bounds = (x1, y1, x2, y2)

    # Extract coordinates
    x1, y1, x2, y2 = bounds
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return image

def compute_xpath_similarity(xpath1: str, xpath2: str) -> float:
    """
    Compute the similarity of two XPath strings using difflib sequence matching.
    Returns a value in [0,1], where higher similarity is closer to 1.
    """
    seq_matcher = difflib.SequenceMatcher(None, xpath1.replace('//','/'), xpath2.replace('//','/'))
    return round(seq_matcher.ratio(), 4)

def get_element_xpath(tree: ET.ElementTree, element: ET.Element) -> str:
    """
    Get the XPath of an element, using class name + index format

    Args:
        tree: XML element tree
        element: The element to get the XPath for

    Returns:
        XPath string of the element, e.g.: /root/LinearLayout[0]/FrameLayout[1]/TextView[0]
    """
    if element is None:
        return ""
    
    # Get the root node
    root = tree.getroot()
    
    # Build path from root to target element
    path_elements = []
    current = element
    
    # Traverse upward from the current element to the root node
    while current is not None and current != root:
        parent = current.getparent()
        if parent is None:
            break
        
        # Get the class of the current element
        class_name = current.attrib.get('class', '')
        if '.' in class_name:
            # If it's a fully qualified class name, take only the last part
            class_name = class_name.split('.')[-1]
        
        # Find sibling nodes of the same type and compute the index
        siblings = [child for child in parent if child.attrib.get('class', '').endswith(class_name)]
        index = siblings.index(current)
        
        # Add the current node to the path
        path_elements.append(f"{class_name}[{index}]")
        
        # Move to the parent node
        current = parent
    
    # Add the root node
    path_elements.append("root")
    
    # Reverse the path and combine into an XPath string
    path_elements.reverse()
    xpath = "/" + "/".join(path_elements)
    
    return xpath


def compute_bounds_similarity(ori_bounds, target_bounds, img_diag_norm=3000.0):
    """
    Compute the similarity of two bounds (position + size difference), returns [0,1].
    """
    try:
        x1_1, y1_1, x2_1, y2_1 = ori_bounds
        x1_2, y1_2, x2_2, y2_2 = target_bounds
    except:
        return 0.0

    # Distance between center points
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    pos_dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # Size difference
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    size_dist = math.sqrt((w1 - w2) ** 2 + (h1 - h2) ** 2)

    # Total difference
    total_dist = pos_dist + size_dist

    # Convert to similarity (smaller distance means more similar, normalized)
    similarity = 1 - min(total_dist / img_diag_norm, 1.0)
    return round(similarity, 4)


def encode_image_to_base64(image_path_or_obj):
    """
    Encode an image to a base64 string

    Args:
        image_path_or_obj: Image path or PIL Image object

    Returns:
        Base64-encoded string
    """
    import io
    import base64
    
    if isinstance(image_path_or_obj, str):
        # If it's a path, read from file
        with open(image_path_or_obj, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        # If it's a PIL Image object, convert to byte stream
        buffer = io.BytesIO()
        image_path_or_obj.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def draw_original_element_on_image(image, bounds, bg_color=(255, 0, 0), bg_alpha=128):
    """
    Draw original element bounds on the image, using a red border and "original UI component" label

    Args:
        image: PIL Image object or image path
        bounds: Element bounds, in format (x1, y1, x2, y2) or "[x1,y1][x2,y2]"
        bg_color: Background rectangle color (R,G,B), defaults to red (255, 0, 0)
        bg_alpha: Background rectangle opacity 0=fully transparent, 255=opaque, defaults to 128

    Returns:
        The annotated image object
    """

    image = image.copy()  # Prevent modifying the original image

    # Parse bounds
    if isinstance(bounds, str):
        # If in string format "[x1,y1][x2,y2]", parse into coordinates
        import re
        match = re.findall(r"\[(\d+),(\d+)\]", bounds)
        if match and len(match) == 2:
            x1, y1 = int(match[0][0]), int(match[0][1])
            x2, y2 = int(match[1][0]), int(match[1][1])
            bounds = (x1, y1, x2, y2)
    
    # Extract coordinates
    x1, y1, x2, y2 = bounds

    # Compute component size and diagonal length (for dynamic border width adjustment)
    w, h = x2 - x1, y2 - y1
    diag = sqrt(w**2 + h**2)

    # Dynamic border width
    border_width = int(max(2, min(12, diag * 0.02)))

    # Create drawing object
    draw = ImageDraw.Draw(image)

    # Draw rectangle border
    draw.rectangle([x1, y1, x2, y2], outline=bg_color, width=border_width)

    # ===== Add text label =====
    label = "original UI component"
    
    # Dynamic font size (scales with box size)
    font_size = int(max(16, min(32, diag * 0.08)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Place text above the rectangle, centered
    text_x = x1 + (w - text_w) // 2
    text_y = max(0, y1 - text_h - 6)  # Avoid going beyond the top of the image

    # If there's not enough space above, place inside the rectangle at the top
    if text_y < 0:
        text_y = y1 + 6
        text_x = x1 + (w - text_w) // 2
    
    # Background box (semi-transparent)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    padding = 4
    bg_coords = [
        text_x - padding, text_y - padding,
        text_x + text_w + padding, text_y + text_h + padding
    ]
    overlay_draw.rectangle(bg_coords, fill=(*bg_color, bg_alpha))  # Use custom color and opacity
    image = Image.alpha_composite(image, overlay)

    # Draw text (white)
    draw = ImageDraw.Draw(image)
    draw.text((text_x, text_y), label, fill="white", font=font)

    return image

def draw_replay_element_on_image(image, bounds, id, bg_color=(0, 128, 0), bg_alpha=160):
    """
    Draw new version element bounds on the image, using a green border and sequence number label

    Args:
        image: PIL Image object
        bounds: Element bounds, in format (x1, y1, x2, y2) or "[x1,y1][x2,y2]"
        id: Element ID number
        bg_color: Background rectangle color (R,G,B), defaults to green (0, 128, 0)
        bg_alpha: Background rectangle opacity 0=fully transparent, 255=opaque, defaults to 160

    Returns:
        The annotated image object
    """
    image = image.copy()  # Prevent modifying the original image
    # Parse bounds
    if isinstance(bounds, str):
        # If in string format "[x1,y1][x2,y2]", parse into coordinates
        import re
        match = re.findall(r"\[(\d+),(\d+)\]", bounds)
        if match and len(match) == 2:
            x1, y1 = int(match[0][0]), int(match[0][1])
            x2, y2 = int(match[1][0]), int(match[1][1])
            bounds = (x1, y1, x2, y2)
    
    # Extract coordinates
    x1, y1, x2, y2 = bounds

    # Compute component size and diagonal length (for dynamic border width adjustment)
    w, h = x2 - x1, y2 - y1
    diag = sqrt(w**2 + h**2)

    # Dynamic border width
    border_width = int(max(2, min(8, diag * 0.02)))

    # Create drawing object
    draw = ImageDraw.Draw(image)

    # Draw rectangle border
    draw.rectangle([x1, y1, x2, y2], outline=bg_color, width=border_width)  # Use custom color

    # Label processing
    label = str(id)
    
    # Dynamic font size
    font_size = int(max(18, min(48, diag * 0.18)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Smart label position selection
    image_width, image_height = image.size
    
    # Compute the ratio of label area to component area
    label_area = (text_w + 8) * (text_h + 8)
    component_area = w * h
    occlusion_ratio = label_area / component_area if component_area > 0 else 1
    
    # Determine whether to place inside or outside
    min_area_threshold = 10000  # Minimum area threshold
    max_occlusion_ratio = 0.15  # Maximum occlusion ratio (15%)
    
    use_external_position = (
        component_area < min_area_threshold or 
        occlusion_ratio > max_occlusion_ratio or
        w < text_w + 16 or  # Component width too small for label
        h < text_h + 16     # Component height too small for label
    )
    
    if use_external_position:
        # External position: outside the top-right corner of the component box
        text_x = x2 + 4
        text_y = y1
        
        # Boundary check
        if text_x + text_w > image_width:
            # Right side exceeds boundary, try placing on the left
            text_x = x1 - text_w - 4
            if text_x < 0:
                # Left side also exceeds boundary, try placing above
                text_x = x1
                text_y = y1 - text_h - 4
                if text_y < 0:
                    # Not enough space above either, place below
                    text_x = x1
                    text_y = y2 + 4
                    if text_y + text_h > image_height:
                        # Final fallback to internal top-right corner
                        text_x = x2 - text_w - 4 if (x2 - x1) >= text_w + 8 else x1 + 4
                        text_y = y1 + 4
    else:
        # Internal position: top-right corner inside the component box
        text_x = x2 - text_w - 4 if (x2 - x1) >= text_w + 8 else x1 + 4
        text_y = y1 + 4
    
    # Final boundary check
    if text_y + text_h > image_height:
        text_y = image_height - text_h - 4
    if text_y < 0:
        text_y = 4
    if text_x < 0:
        text_x = 4
    if text_x + text_w > image_width:
        text_x = image_width - text_w - 4
    
    # Background rectangle coordinates
    padding = 4
    bg_coords = [
        text_x - padding, 
        text_y - padding, 
        text_x + text_w + padding, 
        text_y + text_h + padding
    ]
    
    # If the image is in RGBA mode, use a semi-transparent background
    if image.mode == "RGBA":
        # Semi-transparent background
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(bg_coords, fill=(*bg_color, bg_alpha))  # Use custom color and opacity
        image = Image.alpha_composite(image, overlay)
        
        # Draw text (white)
        draw = ImageDraw.Draw(image)
    else:
        # Semi-transparency not supported, draw solid background directly
        draw.rectangle(bg_coords, fill=bg_color)  # Use custom color

    # Draw text (white)
    draw.text((text_x, text_y), label, fill="white", font=font)
    
    return image

def get_encoded_image(image, model_type="gpt"):
    """
    Encode a PIL Image object into a format suitable for different LLM model APIs

    Args:
        image: PIL Image object
        model_type: Model type, supports "gpt", "qwen", "claude"

    Returns:
        Image data dictionary suitable for the specified model
    """
    import io
    import base64

    
    base64_image = encode_image_to_base64(image)
    
    # Determine image type, defaults to PNG
    image_type = "png"
    
    # Return different formats based on model type
    if "gpt" in model_type.lower() or "o4-mini" in model_type.lower() or "deepseek" in model_type.lower() or "qwen" in model_type.lower():
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}
        }
    elif "qwen" in model_type.lower():  # Alternative Qwen format
        return {
            "type": "image",
            "image": f"data:image;base64,{base64_image}"
        }
    elif "claude" in model_type.lower():
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{image_type}",
                "data": base64_image
            }
        }
    else:
        # Default format
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}
        }


def openai_chat(system_prompt, user_prompt, api_key, model_name, model_type):
    # client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1/")
    client = OpenAI(api_key=api_key, timeout=30.0)

    if 'deepseek' in model_type:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0)
        # client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1/")

    
    if 'o4-mini' in model_name or 'gpt-5' in model_name or 'gpt-4.1-mini' in model_name:
        temperature = 0.8

    # find documents
    if system_prompt:
        completions = client.beta.chat.completions.parse(
            model=model_name,
            temperature=temperature,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt['ori_analyze']
                },
                {
                    "role": "user",
                    "content": user_prompt['update_analyze']
                }
            ]
        )
        results = ''
        for key, completion in enumerate(completions.choices):
            key += 1
            result = completion.message.content
            results += result

    token_usage = {}
    usage = completions.usage
    token_usage['base_model'] = model_name
    token_usage['prompt_tokens'] = usage.prompt_tokens
    token_usage['completion_tokens'] = usage.completion_tokens
    token_usage['total_tokens'] = usage.total_tokens

    return results, token_usage  # Return a dict of results

def get_find_result(analyze_result: str) -> str:
    """
    Parse the analysis result for page found, returns YES or NO.
    """
    find_result = ''

    try:
        # Match YES or NO following the "Your Answer" section (supports newlines and Markdown format)
        m = re.search(
            r'(?is)^\s*(?:#+\s*)?Your\s+Answer\b[\s:：]*([\s\S]*?)\b(YES|NO)\b',
            analyze_result,
            re.M
        )
        if m:
            # Captured result
            answer_block = m.group(1) + m.group(2)
            # Prioritize YES
            if re.search(r'\bYES\b', answer_block, re.I):
                find_result = 'YES'
            elif re.search(r'\bNO\b', answer_block, re.I):
                find_result = 'NO'
    except Exception:
        find_result = ''

    return find_result



def get_component_no(text):
    """
    Extract component number from LLM response.
    """
    # First try to match bracket content in the line after "### Matched_UI_No"
    match = re.search(r'### Matched_UI_No\s*\n\s*\[(.*?)\]', text)

    if match:
        result = match.group(1)
        print(result)
        return result
    else:
        # Fallback: find content inside the last []
        matches = re.findall(r'\[(.*?)\]', text)
        if matches:
            result = matches[-1]  # Take the last one
            print(result)
            return result

    print("No matching content found")
    return ''

def post_process_html_report(sample_path):
    """
    Add four additional columns after the summary table:
    If the result is success, compare whether the Resource ID, Text, Content Desc, and Class
    of the Original Element and Matched Element are the same. If they match, True; otherwise, False.
    - Resource ID True or False
    - Text True or False
    - Content Desc True or False
    - Class True or False
    """


    html_report_path = os.path.join(sample_path, "matching_report.html")
    if not os.path.exists(html_report_path):
        print(f"HTML report file does not exist: {html_report_path}")
        return
        
    with open(html_report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the summary table
    summary_table = soup.select_one('.summary table')
    if not summary_table:
        print("Summary table not found")
        return
    
    # Add new header columns
    header_row = summary_table.select_one('tr')
    header_row.append(soup.new_tag('th'))
    header_row.contents[-1].string = 'Resource ID Match'
    header_row.append(soup.new_tag('th'))
    header_row.contents[-1].string = 'Text Match'
    header_row.append(soup.new_tag('th'))
    header_row.contents[-1].string = 'Content Desc Match'
    header_row.append(soup.new_tag('th'))
    header_row.contents[-1].string = 'Class Match'
    
    # Find all test cases
    case_containers = soup.select('.case-container')
    
    # Create a dictionary to store attribute matching results for each test case
    case_results = {}
    
    # Iterate through each test case and extract attribute comparison results
    for case in case_containers:
        # Get test case identification information
        header = case.select_one('.case-header')
        if not header:
            continue
            
        record_app = header.get('data-record-app')
        replay_app = header.get('data-replay-app')
        run_count_elem = header.select_one('h2')
        
        # Extract run_count
        run_count = None
        if run_count_elem:
            run_match = re.search(r'Run (\d+)', run_count_elem.text)
            if run_match:
                run_count = run_match.group(1)
        
        if not (record_app and replay_app and run_count):
            continue
            
        case_id = (record_app, replay_app, run_count)
        
        # Check if matching was successful
        result_span = header.select_one('.success, .failure')
        is_success = result_span and 'success' in result_span.get('class', [])
        
        if not is_success:
            # If matching failed, all attribute comparisons are False
            case_results[case_id] = {
                'resource_id': False,
                'text': False,
                'content_desc': False,
                'class': False
            }
            continue
        
        # Get attributes of the original element and the matched element
        original_attrs = case.select('.column')[0].select('.attribute p') if len(case.select('.column')) > 0 else []
        matched_attrs = case.select('.column')[1].select('.attribute p') if len(case.select('.column')) > 1 else []
        
        # Extract attribute values
        original_values = {}
        matched_values = {}
            
        for attr in original_attrs:
            if 'Resource ID:' in attr.text:
                # Use a more precise method to extract the value
                parts = attr.text.split('Resource ID:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    original_values['resource_id'] = '' if value == 'N/A' or value == "" else value
            elif 'Text:' in attr.text:
                parts = attr.text.split('Text:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    original_values['text'] = '' if value == 'N/A' or value == "" else value
            elif 'Content Desc:' in attr.text:
                parts = attr.text.split('Content Desc:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    original_values['content_desc'] = '' if value == 'N/A' or value == "" else value
            elif 'Class:' in attr.text:
                parts = attr.text.split('Class:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    original_values['class'] = '' if value == 'N/A' or value == "" else value
        
            
        for attr in matched_attrs:
            if 'Resource ID:' in attr.text:
                parts = attr.text.split('Resource ID:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    matched_values['resource_id'] = '' if value == 'N/A' or value == "" else value
            elif 'Text:' in attr.text:
                parts = attr.text.split('Text:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    matched_values['text'] = '' if value == 'N/A' or value == "" else value
            elif 'Content Desc:' in attr.text:
                parts = attr.text.split('Content Desc:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    matched_values['content_desc'] = '' if value == 'N/A' or value == "" else value
            elif 'Class:' in attr.text:
                parts = attr.text.split('Class:', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    matched_values['class'] = '' if value == 'N/A' or value == "" else value
        
        # Initialize the result dictionary for the current case_id
        case_results[case_id] = {}
        
        # Compare attribute values
        if original_values.get('resource_id') == '' and matched_values.get('resource_id') == '':
            case_results[case_id]['resource_id'] = 'NA'
        else:
            case_results[case_id]['resource_id'] = original_values.get('resource_id') == matched_values.get('resource_id')

        if original_values.get('text') == '' and matched_values.get('text') == '':
            case_results[case_id]['text'] = 'NA'
        else:
            case_results[case_id]['text'] = original_values.get('text') == matched_values.get('text')

        if original_values.get('content_desc') == '' and matched_values.get('content_desc') == '':
            case_results[case_id]['content_desc'] = 'NA'
        else:
            case_results[case_id]['content_desc'] = original_values.get('content_desc') == matched_values.get('content_desc')

        if original_values.get('class') == '' and matched_values.get('class') == '':
            case_results[case_id]['class'] = 'NA'
        else:
            case_results[case_id]['class'] = original_values.get('class') == matched_values.get('class')

        
        # Debug output - attribute values for each test case
        # print(f"Case {case_id}:")
        # print(f"  original_values: {original_values}")
        # print(f"  matched_values: {matched_values}")
        # print(f"  match results: {case_results[case_id]}")

    
    # Update each row in the summary table
    data_rows = summary_table.select('tr')[1:]  # Skip the header row
    
    for row in data_rows:
        cells = row.select('td')
        if len(cells) < 5:
            continue
            
        record_app = cells[0].text.strip()
        replay_app = cells[1].text.strip()
        run_count = cells[2].text.strip()
        result = cells[3].text.strip()
        
        case_id = (record_app, replay_app, run_count)
        
        # If it's a failure case or no matching test case result found, add False
        if result == 'Failure' or case_id not in case_results:
            for _ in range(4):
                row.append(soup.new_tag('td'))
                row.contents[-1].string = 'NA'
                row.contents[-1]['class'] = ['failure']
        else:
            # Add resource ID matching result
            row.append(soup.new_tag('td'))
            row.contents[-1].string = str(case_results[case_id]['resource_id'])
            row.contents[-1]['class'] = ['success'] if case_results[case_id]['resource_id'] else ['failure']
            
            # Add text matching result
            row.append(soup.new_tag('td'))
            row.contents[-1].string = str(case_results[case_id]['text'])
            row.contents[-1]['class'] = ['success'] if case_results[case_id]['text'] else ['failure']
            
            # Add content description matching result
            row.append(soup.new_tag('td'))
            row.contents[-1].string = str(case_results[case_id]['content_desc'])
            row.contents[-1]['class'] = ['success'] if case_results[case_id]['content_desc'] else ['failure']
            
            # Add class matching result
            row.append(soup.new_tag('td'))
            row.contents[-1].string = str(case_results[case_id]['class'])
            row.contents[-1]['class'] = ['success'] if case_results[case_id]['class'] else ['failure']
    
    # Update CSS styles, add styles for the new columns
    style_tag = soup.select_one('style')
    if style_tag:
        new_style = """
                .match-true {
                    color: green;
                    font-weight: bold;
                }
                .match-false {
                    color: red;
                    font-weight: bold;
                }
        """
        style_tag.append(new_style)
    
    # Generate the new HTML file
    processed_html_path = os.path.join(sample_path, "matching_report_processed.html")
    with open(processed_html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"Processed HTML report generated: {processed_html_path}")

if __name__ == "__main__":
    parent_dir = "/Users/ssw/Desktop/yiheng"
    # Get all subdirectories below
    # for app_name in os.listdir(parent_dir):
        
    #     if os.path.isdir(os.path.join(parent_dir, app_name)):
    #         print(f"Processing: {app_name}")
    #         post_process_html_report(os.path.join(parent_dir, app_name))
    
    post_process_html_report("/Users/ssw/Downloads/html_report")
    
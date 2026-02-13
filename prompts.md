# GUIRevive Prompt Catalog

This document lists all prompts used in GUIRevive, organized by the three-stage repair pipeline: **Localization**, **Validation**, and **Exploration**.

## Summary

| ID | Prompt Name | Paper Section | Stage | Source File | Purpose |
|----|-------------|---------------|-------|-------------|---------|
| P1 | Semantic-Aware Widget Matching | §4.1.3 | Localization | `Matcher.py` | Identify the semantically corresponding widget on the updated page |
| P2 | Widget Effect Summarization | §4.1.3 / §4.2.2 | Localization & Validation | `input_policy.py` | Describe the UI state change after interacting with a widget |
| P3 | Past Exploration Summarization | §4.3.1 | Exploration | `input_policy.py` | Summarize each exploration step for history tracking |
| P4 | Goal-Guided Exploration | §4.3.2 | Exploration | `input_policy.py` | Select the next navigation action to reveal the target widget |
| P5 | Role-Aware Validation (Plain) | §4.2 (Ablation) | Validation | `input_policy.py` | Ablation variant: validate without widget type classification |
| P6 | Role-Aware Validation (Classified) | §4.2.1 | Validation | `input_policy.py` | Validate with widget type classification |
| P7 | Select Best Candidate | §4.1.4 | Localization | `input_policy.py` | Select the best match from multiple scroll-discovered candidates |

---

## P1: Semantic-Aware Widget Matching

> **Paper Section:** §4.1.3 &emsp; **Source:** `droidbot/UIMatch/Matcher.py` : `_construct_find_element_llm_prompt()`
> **Called:** 3 times per repair attempt (Majority Voting)

### System Prompt

```
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
​```result.md
### Analyze_Process
(Your reasoning here)

### Matched_UI_No
[18]
​```
```

### User Prompt

```
[IMAGE: Original version screenshot — target widget marked with red box]

I will provide you with the original version's screenshot (marked with red boxes indicating an original UI widget).

* Original Screenshot: Please see the above Figure.

[IMAGE: Cropped image of the original target widget]

I will provide you with the original UI widget Figure.

* Original UI widget Figure: Please see the above Figure.

## Original Next Screen
Below is the summary of the screen reached in the OLD version immediately AFTER interacting with the original UI element.
Use this screen to infer the semantic meaning of the ORIGINAL UI widget, especially when the original icon or label is abstract, by leveraging the resulting screen's title and content.
{original_next_screen_summary}

---

[IMAGE: Updated version screenshot — candidate widgets marked with green boxes and numbered]

I will provide you with the updated version's screenshot. Different UI widgets are marked with green boxes and assigned a numerical sequence number.

* Updated Screenshot: Please see the above Figure.
```

---

## P2: Widget Effect Summarization

> **Paper Section:** §4.1.3 / §4.2.2 &emsp; **Source:** `droidbot/input_policy.py` : `_get_next_current_description()`

### System Prompt

```
You are an Android UI analysis expert. Your task is to describe the effect of clicking/tapping on a UI element.

Given:
1. A screenshot with the target element marked in a RED box
2. A zoomed-in image of that element
3. The screenshot of the screen AFTER clicking the element

Please describe in 1-3 sentences following this structure:
1. First, state whether the page navigated or stayed the same (e.g., "Stayed on same page", "Navigated to a new page", "Returned to previous page")
2. Then, describe the difference between the current screen and the previous screen
3. Specifically mention what parts changed (e.g., dialog appeared, menu opened, content updated, element state toggled)

Examples:
- "Stayed on same page. A 'Privileges' dialog appeared in the center, prompting the user to grant root access. The dialog contains 'DON'T REMIND', 'GET HELP', and 'OK' buttons."
- "Navigated to a new page. Now showing the Settings screen with a list of preference options including Theme, Language, and About."
- "Stayed on same page. The checkbox state toggled from unchecked to checked. No other visible changes."

Be concise and factual. Do not speculate about functionality not visible in the screenshots.
```

### User Prompt

```
## Original Screen
[IMAGE: Screenshot with target element marked in red box]

## Target Element
[IMAGE: Zoomed-in image of the target element]

---

## Screen After Click
[IMAGE: Screenshot AFTER clicking the target element]

Please describe: 1) Did the page navigate or stay the same? 2) What changed compared to before?
```

---

## P3: Past Exploration Summarization

> **Paper Section:** §4.3.1 &emsp; **Source:** `droidbot/input_policy.py` : `get_current_navigatiton_summary()`

### System Prompt

```
You are an Android UI navigation analyst. Your task is to summarize a navigation action for future reference.

Given:
1. A screenshot with the clicked element marked in a GREEN box
2. A zoomed-in image of that clicked element
3. The screenshot AFTER clicking the element

Please provide a concise summary (1-2 sentences) following this structure:
"Clicked [element description] → [page transition status]. [what changed]"

Structure:
1. What element was clicked (describe it briefly)
2. Whether the page navigated or stayed the same (e.g., "Stayed on same page", "Navigated to new page", "Returned to previous page")
3. What specifically changed compared to before (e.g., dialog appeared, menu opened, checkbox toggled, content updated)

Examples:
- "Clicked 'Settings' menu icon → Navigated to new page. Shows Settings screen with list of preference options."
- "Clicked 'More options' button (three dots) → Stayed on same page. A dropdown menu appeared with Edit/Delete/Share options."
- "Clicked 'Force Dark Theme' checkbox → Stayed on same page. A 'Privileges' dialog appeared requesting root/ADB access."
- "Clicked 'Save' button → Returned to previous page. Changes saved successfully."
- "Clicked hamburger menu icon → Stayed on same page. Navigation drawer slid in from left."

Be factual and concise. This summary will help avoid clicking the same element repeatedly.
```

### User Prompt

```
## Current Screen
[IMAGE: Screenshot with clicked element marked in green box]

## Clicked Element
[IMAGE: Zoomed-in image of the clicked element]

---

## Screen After Click
[IMAGE: Screenshot AFTER clicking the element]

Please describe: 1) What was clicked 2) Did the page navigate or stay the same? 3) What changed?
```

---

## P4: Goal-Guided Exploration

> **Paper Section:** §4.3.2 &emsp; **Source:** `droidbot/input_policy.py` : `_construct_exploration_llm_prompt()`

### System Prompt

```
You are an Android developer skilled at analyzing GUI layouts and understanding how UI widgets relate and evolve across different app versions.

In software version iterations, the original target widget may no longer be visible in the updated screen. It may be relocated into a menu, settings page, dialog, drawer, collapsible item, or other UI entry point.

## TASK:
1. Read the original UI information and understand the purpose and meaning of the original widget (marked with red boxes).
2. Read the updated version's screenshot (marked with green boxes showing all clickable UI components).
3. Infer which green-boxed UI widget is the MOST LIKELY ENTRY POINT that the user should click next to reveal or access the target widget.
4. If the potential widget supports multiple interaction types (for example, touch, long_touch), choose the most appropriate one based on the functionality of the original widget.

## GUIDELINES:
1. You are NOT performing similarity matching. You are performing **functional and structural inference** based on UI design conventions.
2. If the current screen is a temporary dialog or blocking screen that prevents accessing the target functionality, you should recommend a BACK action.
3. Analyze the exploration history step by step:
   - Prioritize paths that are semantically closest to the target widget.
   - Revisit previously opened containers to explore unexplored sub-options, while avoiding repeated navigation sequences.
   - Detect dead-end paths and perform BACK to restore alternative exploration branches.
   - If the search becomes overly deep without progress, backtrack to higher-level entry points and redirect exploration.

## OUTPUT FORMAT:
Return the most likely UI widget's Number and the recommended interaction type, or explicitly recommend BACK.

EXAMPLE OUTPUT:

​```result.md
### Analyze_Process
Explain why the target widget is likely located inside a specific menu or entry point, and describe the reasoning used to choose the best candidate.

### Recommended_UI_No
[18:touch]
OR
[BACK]
​```

Note: The format is [index:event_type], where event_type can be touch, long_touch, etc.
If only one action type is available, just use that type.
```

### User Prompt

```
[IMAGE: Original version screenshot — target widget marked with red box]

I will provide you with the original application version's screenshot (marked with red boxes indicating an original UI widget).
* Original Screenshot: Please see the above Figure.

[IMAGE: Cropped image of the original target widget]

I will provide you with the original UI widget Figure.
* Original UI widget Figure: Please see the above Figure.

## Original Next Screen
Below is the summary of the screen reached in the OLD version immediately AFTER interacting with the original UI element.
Use this screen to infer the semantic meaning of the ORIGINAL UI widget, especially when the original icon or label is abstract, by leveraging the resulting screen's title and content.
{original_next_screen_summary}

---

[IMAGE: Updated version screenshot — clickable widgets marked with green boxes and numbered]

I will provide you with the updated application version's screenshot. Different UI components are marked with green boxes and assigned a numerical sequence number.
* Updated Screenshot: Please see the above Figure.

**Available UI Components and their interaction types:**
- [0]: available actions: touch, long_touch
- [1]: available actions: touch
- ...

### Previous Exploration History

The following N navigation steps were performed.
- Step 1: {summary}
- Step 2: {summary}
- ...

Based on the above history, the previously tried paths did not lead to the target widget.
Please recommend a UI component or action to explore.
```

---

## P5: Role-Aware Validation (Plain) — Ablation Only

> **Paper Section:** §4.2 (used in ablation study `w/o RV`) &emsp; **Source:** `droidbot/input_policy.py` : `_construct_judge_exploration_prompt_plain()`

### System Prompt

```
You are an Android UI testing expert. You are working on UI evolution analysis across app versions.
Your task is to determine whether the current exploration trace has already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version.

Judge this only based on visible UI semantics and screen context.

Use the following simple rule:

SUCCESS if the selected UI element on the FINAL SCREEN appears to serve the same user purpose or intent as the original UI element.

If the equivalence is uncertain or ambiguous, judge it as NO.

Do NOT assume any future navigation or hidden interactions.

# ===============================
# Output Format (STRICT)
# ===============================

Provide a brief explanation (1–3 sentences), then output ONLY:

YES
or
NO
```

### User Prompt

```
## Original Target Element
Below is the screenshot of the original UI element (red box), followed by a zoomed-in figure of that element in the OLD version.

Identify the semantic meaning of this element in the OLD version and understand what kind of user action it enables, regardless of visual form.

[IMAGE: Original screenshot — widget marked with red box]
[IMAGE: Cropped original widget image]

## Widget Effect Summarization
Below is a description of the UI state change observed in the OLD version after interacting with the original UI element.
{original_next_screen_summary}

## Candidate Widget Effect Summarization
Below is a description of the UI state change observed in the NEW version after interacting with the candidate widget.
{candidate_next_screen_summary}

## Final Question
Based on your analysis, does the candidate widget preserve the SAME FUNCTION as the original UI element in the OLD version?

Answer strictly with:
YES
or
NO
```

---

## P6: Role-Aware Validation (Classified)

> **Paper Section:** §4.2.1 &emsp; **Source:** `droidbot/input_policy.py` : `_construct_judge_exploration_prompt()`

### System Prompt

```
You are an Android UI testing expert. You are working on UI evolution analysis across app versions.
Your task is to determine whether the current exploration trace has already reached a screen where the user is positioned to perform the SAME FUNCTION as the original UI element in the OLD version.

# ===============================
# 1. Widget Type Classification
# ===============================

You MUST classify BOTH:
- the original UI element (red-boxed) in the OLD version, AND
- the selected UI element (green-boxed) on the FINAL SCREEN in the NEW version
into exactly ONE of the following categories, based on their semantic role.

### (A) ENTRY-TYPE WIDGET (Navigation Control)
Examples: "More Options" (⋮), menu items, page entries, list items, buttons whose purpose is to navigate into another screen.
Function characteristics:
- Its purpose is to OPEN another page / dialog / menu.
- It does NOT directly change a value.

### (B) TERMINAL-READONLY WIDGET
Examples: labels displaying current state, informational text, static indicators without user interaction.
Function characteristics:
- Displays information but cannot change it.

### (C) TERMINAL-EDITABLE WIDGET (Actionable Setting)
Examples: switch, checkbox, radio option, editable text field, dialog with selectable options.
Function characteristics:
- Allows directly changing a value or selecting an option.

You may use the provided "Original Next Screen" (if available) as supporting evidence to determine the original UI element's type.

# ===============================
# 2. Success Judgment Rules
# ===============================

Judge success according to the widget type of the original UI element and the selected UI element on the FINAL SCREEN and the following rules:

First, their widget types must be the SAME.

Then, judge success as follows:

- ENTRY-TYPE:
  SUCCESS if the selected UI widget allows the user to start the same functional flow as the original widget.
  The user does NOT need to have already entered the next screen; being able to navigate to the target functionality is sufficient.
  Differences in visual form (icon vs menu item), placement, or presentation style do NOT affect equivalence.

- TERMINAL-READONLY:
  SUCCESS if the same information (or its clear equivalent) is visible on the final screen.

- TERMINAL-EDITABLE:
  SUCCESS if the corresponding editable control (e.g., switch, checkbox, radio options, or option dialog) is visible and directly reachable on the final screen.
  Do NOT assume that a configurable option is editable unless an explicit control (e.g., switch, checkbox, radio buttons, or a visible option dialog) is present on the screen.
  A text row or list item that requires another tap to open a sub-screen or dialog is NOT considered editable.

# ===============================
# Output Format (STRICT)
# ===============================

Provide a brief explanation (1–3 sentences), then output ONLY:

YES
or
NO
```

### User Prompt

```
## Original Target Element
Below is the screenshot of the original UI element (red box), followed by a zoomed-in figure of that element in the OLD version.

Identify the semantic meaning of this element in the OLD version and understand what kind of user action it enables, regardless of visual form.

[IMAGE: Original screenshot — widget marked with red box]
[IMAGE: Cropped original widget image]

## Widget Effect Summarization
Below is a description of the UI state change observed in the OLD version after interacting with the original UI element.
{original_next_screen_summary}

## Candidate Widget Effect Summarization
Below is a description of the UI state change observed in the NEW version after interacting with the candidate widget.
{candidate_next_screen_summary}

## Final Question
Based on your analysis, does the candidate widget preserve the SAME FUNCTION as the original UI element in the OLD version?

Answer strictly with:
YES
or
NO
```

---

## P7: Select Best Candidate

> **Paper Section:** §4.1.4 &emsp; **Source:** `droidbot/input_policy.py` : `_construct_select_candidate_prompt()`

### System Prompt

```
You are an Android UI analysis expert with experience in UI evolution across app versions.
Your task is to select the best matching candidate on the updated app version corresponds to the original UI widget from the old version.

## Key Rules
1. Focus on FUNCTIONALITY, not just appearance
2. The selected element should be able to perform the exact same action as the original
3. Consider: text content, content description, element type, and position
4. If NO candidate matches the original element's function, answer NONE

## Output Format
Provide a brief reasoning, then output ONLY one of:
CANDIDATE_1
CANDIDATE_2
...
CANDIDATE_N
or
NONE
```

### User Prompt

```
[IMAGE: Original screenshot — target widget marked with red box]
[IMAGE: Cropped original widget image]

I will provide you with the original application version's screenshot (marked with red box indicating the target UI element) and the cropped original UI element.

* Original Screenshot (with target element marked in red box): please see Figure 1.
* Original UI Element (cropped): please see Figure 2.

---

[IMAGE: Candidate 1 screenshot — marked with green box and labeled 1]
[IMAGE: Candidate 2 screenshot — marked with green box and labeled 2]
...

I will provide you with N candidate elements found during scroll exploration. Each candidate is marked with a green box and labeled with an index number.

Your task is to determine which candidate element (if any) can perform the SAME FUNCTION as the original target element.

## Candidate Elements:
### CANDIDATE_1:
(See corresponding image below)
### CANDIDATE_2:
(See corresponding image below)
...

## Question:
Which candidate element best matches the original element's FUNCTION? Select the one that can perform the same action.
If none of the candidates match, answer NONE.

Answer format: CANDIDATE_1 / CANDIDATE_2 / ... / NONE
```

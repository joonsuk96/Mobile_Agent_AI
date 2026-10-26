"""Model-input-only compaction. Checkpoints and observation evidence stay intact."""
import json
from langchain.messages import ToolMessage

IMAGE_TYPES = {"image_url", "input_image", "image"}

def compact_visual_history(messages, keep=2):
    images = [i for i, message in enumerate(messages) if isinstance(message, ToolMessage)
              and isinstance(message.content, list)
              and any(isinstance(block, dict) and block.get("type") in IMAGE_TYPES for block in message.content)]
    remove = set(images[:-keep]) if keep else set(images)
    if not remove:
        return messages
    compact = []
    for index, message in enumerate(messages):
        if index not in remove:
            compact.append(message)
            continue
        text_blocks = [block for block in message.content if not isinstance(block, dict) or block.get("type") not in IMAGE_TYPES]
        compact.append(message.model_copy(update={"content": text_blocks or "이전 화면 이미지는 생략했어. 기록된 관찰·후보 근거를 참고해."}))
    return compact


def compact_observation_history(messages, keep_recent=4):
    """Drop duplicated *old* runtime context, preserving facts, actions and UI text."""
    tool_indices = [i for i, message in enumerate(messages) if isinstance(message, ToolMessage)]
    old = set(tool_indices[:-keep_recent]) if keep_recent else set(tool_indices)
    if not old:
        return messages

    def clean_screen(value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if "snapshot_id" in result and "elements" in result:
            for key in ("task_progress", "workflow_guidance", "recent_actions", "navigation_recovery", "search_cycle"):
                result.pop(key, None)
        for key in ("screen", "result", "observation"):
            if isinstance(result.get(key), dict):
                result[key] = clean_screen(result[key])
        return result

    def clean_text(text):
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return text
        cleaned = clean_screen(payload)
        return json.dumps(cleaned, ensure_ascii=False) if cleaned != payload else text

    result = []
    for index, message in enumerate(messages):
        if index not in old:
            result.append(message)
            continue
        content = message.content
        if isinstance(content, str):
            content = clean_text(content)
        elif isinstance(content, list):
            content = [{**block, "text": clean_text(block["text"])}
                       if isinstance(block, dict) and block.get("type") == "text" else block for block in content]
        result.append(message.model_copy(update={"content": content}))
    return result

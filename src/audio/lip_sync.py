def generate_lip_sync(text: str):
    phonemes = []
    for char in text.lower():
        if char in "aeiou":
            phonemes.append({"time": 0.1, "mouth_open": 0.8})
        elif char in "bcdfghjklmnpqrstvwxyz":
            phonemes.append({"time": 0.05, "mouth_open": 0.2})
        else:
            phonemes.append({"time": 0.05, "mouth_open": 0.0})
    return phonemes

def split_text_for_streaming(text: str, min_length: int = 10, max_length: int = 40) -> list[str]:
    import re

    # Split by major punctuation: 。！？ or newlines
    # Also optionally split by minor punctuation: 、… if the current segment is long enough
    segments = re.split(r'([。！？!?\n])', text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(segments), 2):
        segment = segments[i]
        delimiter = segments[i + 1] if i + 1 < len(segments) else ""
        combined = segment + delimiter

        if not combined.strip():
            continue

        # If combined has minor punctuation, try to split further if it's long
        if len(current_chunk) + len(combined) > max_length:
            sub_segments = re.split(r'([、…])', combined)
            for j in range(0, len(sub_segments), 2):
                sub_segment = sub_segments[j]
                sub_delimiter = sub_segments[j+1] if j+1 < len(sub_segments) else ""
                sub_combined = sub_segment + sub_delimiter
                
                if len(current_chunk) + len(sub_combined) >= min_length:
                     chunks.append((current_chunk + sub_combined).strip())
                     current_chunk = ""
                else:
                    current_chunk += sub_combined
        else:
            current_chunk += combined
            if any(p in combined for p in "。！？!?\n") or len(current_chunk) >= max_length:
                if len(current_chunk.strip()) >= 1:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

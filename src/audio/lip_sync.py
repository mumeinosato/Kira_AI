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

def split_text_for_streaming(text: str, max_length: int = 50) -> list[str]:
    import re

    segments = re.split(r'([。、\n])', text)
    chunks = []
    current_chunk = ""

    for i in range(0, len(segments), 2):
        segment = segments[i]
        delimiter = segments[i + 1] if i + 1 < len(segments) else ""

        combined = segment + delimiter

        if len(current_chunk) + len(combined) <= max_length:
            current_chunk += combined
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = combined

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

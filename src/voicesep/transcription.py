import gc


def transcribe_tracks(paths, language=None, model_name="small", cancelled=lambda: False):
    """Load ASR after separation. No mandatory API or automatic English-only fallback."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    transcript = []
    try:
        for speaker_id, path in paths:
            if cancelled():
                raise InterruptedError("Transcription cancelled")
            segments, info = model.transcribe(str(path), language=language, vad_filter=True,
                                               beam_size=5, condition_on_previous_text=False)
            for segment in segments:
                if cancelled():
                    raise InterruptedError("Transcription cancelled")
                transcript.append({"speaker_id": speaker_id, "start": segment.start,
                                   "end": segment.end, "text": segment.text.strip(),
                                   "language": info.language})
    finally:
        del model
        gc.collect()
    return sorted(transcript, key=lambda item: (item["start"], item["speaker_id"]))


def subtitle_text(segments, names, kind="srt"):
    def stamp(seconds):
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        sec, ms = divmod(remainder, 1000)
        separator = "." if kind == "vtt" else ","
        return f"{hours:02}:{minutes:02}:{sec:02}{separator}{ms:03}"
    lines = ["WEBVTT", ""] if kind == "vtt" else []
    for index, segment in enumerate(segments, 1):
        lines.extend([str(index), f"{stamp(segment['start'])} --> {stamp(segment['end'])}",
                      f"{names.get(segment['speaker_id'], segment['speaker_id'])}: {segment['text']}", ""])
    return "\n".join(lines)

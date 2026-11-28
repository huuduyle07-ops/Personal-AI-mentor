"""Local speech-to-text for the dashboard microphone, with faster-whisper (OpenAI Whisper on CTranslate2).

Audio never leaves the computer: the model runs on the CPU (int8). The weights are downloaded once from
Hugging Face, only when you press the download button in the dashboard, and are then loaded offline
(local_files_only=True). Whisper detects the language, so English, Korean and Vietnamese all work.

Choosing the language. Whisper's own language guess is reliable for English sentences (0.87-0.99 in
testing) but not for short Korean words, which it spreads over Japanese, French and English (나무 was even
guessed as Vietnamese). So:
  1. a language picked in the dashboard (English / 한국어 / Tiếng Việt) is used as is;
  2. a confident guess (>= CONFIDENT) of English is used directly;
  3. otherwise the clip is transcribed as English, as Korean and, when Whisper gives Vietnamese at least
     VIETNAMESE_CHANCE, as Vietnamese; the transcript Whisper is most confident in (highest average
     log-probability per segment) wins. An empty transcript, one in the wrong script (学教 for 학교 on the
     English pass) or a repetition loop ("quiz me, quiz me, ...") never wins.
Hints ("hotwords") steer the spelling: common requests for English and Korean ("quiz me", not "with me";
퀴즈, not 키즈), plus the deck words for Korean while an active quiz waits (나무, not 다음으로). Each pass is
capped at MAX_TOKENS and makes a single attempt (temperature 0): by default Whisper retries an unsure
pass at higher temperatures, and a pass in the wrong language is always unsure, which took up to 25 s.

Model: WHISPER_MODEL (default "small", about 480 MB; "base" is about 145 MB but much weaker on Korean).
"""
import io
import os
import re
import threading

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODEL = os.environ.get("WHISPER_MODEL", "small")
SIZES = {"tiny": "75 MB", "base": "145 MB", "small": "480 MB", "medium": "1.5 GB"}
MAX_SECONDS = 60
CONFIDENT = 0.85
VIETNAMESE_CHANCE = 0.3
MAX_TOKENS = 100  # about 30-60 s of speech
REPETITIVE = 2.4  # Whisper's own compression-ratio threshold: above it, the text is a repetition loop
HANGUL = re.compile(r"[\u3131-\u318e\uac00-\ud7a3]")
CJK = re.compile(r"[\u3040-\u30ff\u3131-\u318e\u3400-\u9fff\uac00-\ud7a3]")  # kana, Hangul, Chinese


def plausible(text, language):
    """Is this transcript written in the language's own script? A forced pass in the wrong language often
    comes back in another script (학교 heard 'as English' gave 学教), which must not win the comparison."""
    return bool(HANGUL.search(text)) if language == "ko" else not CJK.search(text)
ENGLISH_HINTS = "quiz me, quiz me in Korean, how am I doing, today's total is 20 minutes, I don't know, yes, no, focus"
KOREAN_PHRASES = ("한글로 퀴즈", "퀴즈 내줘", "오늘 공부했어요", "분", "진도", "피곤해요", "바빠요", "내일", "네", "아니요")
MAX_HOTWORDS = 60  # Whisper's prompt is short; keep the hint list well inside it


def decode(wav):
    from faster_whisper import decode_audio
    return decode_audio(io.BytesIO(wav))


def hotwords(words=()):
    """A Korean hint string: common requests plus (up to MAX_HOTWORDS) deck words, no duplicates."""
    unique = list(dict.fromkeys([*KOREAN_PHRASES, *words]))
    return " ".join(unique[:MAX_HOTWORDS])


class SpeechUnavailable(Exception):
    pass


class Transcriber:
    def __init__(self, model=MODEL, loader=None, decoder=decode):
        self.model_name, self.loader, self.decoder = model, loader, decoder
        self._model, self._lock = None, threading.Lock()

    @property
    def size(self):
        return SIZES.get(self.model_name, "a few hundred MB")

    def _load(self, download):
        if self.loader is not None:
            return self.loader(self.model_name, download)
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SpeechUnavailable("Voice input needs the dashboard extra: uv sync --extra dashboard") from None
        try:
            return WhisperModel(self.model_name, device="cpu", compute_type="int8", local_files_only=not download)
        except Exception:  # not downloaded yet (or a broken download)
            if download:
                raise SpeechUnavailable("Downloading the speech model failed. Check your connection and try again.") from None
            return None

    def ready(self):
        """True once the model is on disk (and loaded). Never downloads."""
        with self._lock:
            if self._model is None:
                self._model = self._load(download=False)
            return self._model is not None

    def download(self):
        """One-time download of the model weights, then load them."""
        with self._lock:
            self._model = self._load(download=True)
            return self._model is not None

    def _run(self, audio, language, korean_hints):
        """(text, confidence) for one language; confidence is the mean segment log-probability."""
        hints = korean_hints if language == "ko" else ENGLISH_HINTS if language == "en" else ""
        segments, _ = self._model.transcribe(audio, language=language, beam_size=1, vad_filter=True,
                                             condition_on_previous_text=False, max_new_tokens=MAX_TOKENS,
                                             temperature=0.0,  # one attempt: no slow retries when unsure
                                             **({"hotwords": hints} if hints else {}))
        kept = []
        for segment in segments:
            if segment.start > MAX_SECONDS:
                break
            kept.append(segment)
        text = " ".join(t for t in (segment.text.strip() for segment in kept) if t).strip()
        usable = text and plausible(text, language) and all(s.compression_ratio <= REPETITIVE for s in kept)
        confidence = sum(segment.avg_logprob for segment in kept) / len(kept) if usable else float("-inf")
        return text, confidence

    def transcribe(self, wav, language=None, korean_hints="", allow_vietnamese=True):
        """(text, language) for a short recording (WAV bytes). Empty text if nothing was said.
        `language` fixes the language; otherwise it is chosen as described at the top of this module."""
        if not self.ready():
            raise SpeechUnavailable(f"The speech model isn't downloaded yet ({self.size}).")
        audio = self.decoder(wav)
        candidates = ("en", "ko")
        if language is None:
            probabilities = dict(self._model.detect_language(audio)[2])
            if probabilities.get("en", 0.0) >= CONFIDENT:
                language = "en"
            elif allow_vietnamese and probabilities.get("vi", 0.0) >= VIETNAMESE_CHANCE:
                candidates = ("en", "ko", "vi")
        if language is not None:
            return self._run(audio, language, korean_hints)[0], language
        options = [(*self._run(audio, code, korean_hints), code) for code in candidates]
        text, _, language = max(options, key=lambda option: option[1])
        return text, language

from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:

    def __init__(self):

        self.languages = [
            Language.INDONESIAN,
            Language.ENGLISH,
            Language.CHINESE,
            Language.JAPANESE,
            Language.RUSSIAN,
        ]

        self.detector = (
            LanguageDetectorBuilder
            .from_languages(*self.languages)
            .with_preloaded_language_models()
            .build()
        )

        self.language_map = {
            Language.INDONESIAN: "id",
            Language.ENGLISH: "en",
            Language.CHINESE: "zh",
            Language.JAPANESE: "ja",
            Language.RUSSIAN: "ru",
        }

    def detect(self, text: str) -> dict:

        if not text or not text.strip():

            return {
                "language": "Indonesian",
                "language_code": "id",
                "confidence": 0.0,
                "status": "default"
            }

        text = " ".join(text.strip().split())

        try:

            results = sorted(
                self.detector.compute_language_confidence_values(text),
                key=lambda x: x.value,
                reverse=True
            )

            if not results:

                return self._unknown()

            best = results[0]

            second = (
                results[1]
                if len(results) > 1
                else None
            )

            confidence = float(best.value)

            margin = (
                confidence - float(second.value)
                if second
                else confidence
            )

            language_code = self.language_map.get(
                best.language
            )

            # ------------------------------------------
            # Script-based detection
            # ------------------------------------------

            script_language = self._detect_script(text)

            if script_language:

                return {
                    "language": self._language_name(script_language),
                    "language_code": script_language,
                    "confidence": 1.0,
                    "status": "script"
                }

            # ------------------------------------------
            # Confidence validation
            # ------------------------------------------

            if confidence < 0.40:

                return {
                    "language": "Unknown",
                    "language_code": "unknown",
                    "confidence": round(confidence, 4),
                    "margin": round(margin, 4),
                    "status": "uncertain"
                }

            # ------------------------------------------
            # Ambiguous result
            # ------------------------------------------

            if margin < 0.10:

                return {
                    "language": "Unknown",
                    "language_code": "unknown",
                    "confidence": round(confidence, 4),
                    "margin": round(margin, 4),
                    "status": "ambiguous"
                }

            # ------------------------------------------
            # Valid result
            # ------------------------------------------

            return {
                "language": best.language.name,
                "language_code": language_code,
                "confidence": round(confidence, 4),
                "margin": round(margin, 4),
                "status": "detected"
            }

        except Exception as e:

            print(f"[LanguageDetector] Error: {e}")

            return self._unknown()

    def _detect_script(self, text):

        has_cyrillic = any(
            "\u0400" <= char <= "\u04FF"
            for char in text
        )

        if has_cyrillic:
            return "ru"

        has_hiragana = any(
            "\u3040" <= char <= "\u309F"
            for char in text
        )

        has_katakana = any(
            "\u30A0" <= char <= "\u30FF"
            for char in text
        )

        if has_hiragana or has_katakana:
            return "ja"

        return None

    def _language_name(self, code):

        names = {
            "id": "Indonesian",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ru": "Russian",
        }

        return names.get(code, "Unknown")

    def _unknown(self):

        return {
            "language": "Unknown",
            "language_code": "unknown",
            "confidence": 0.0,
            "status": "unknown"
        }
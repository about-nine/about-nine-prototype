# routes/voice.py - voice STT, TTS, and bio endpoints
from flask import Blueprint, request, jsonify
import base64, tempfile, os
import traceback
import re

from openai import OpenAI
client = OpenAI()

voice_bp = Blueprint("voice", __name__, url_prefix="/api/voice")

MAX_AUDIO_MB = 6


# =========================
# helpers
# =========================

def safe_temp_save(file_storage, mime_type="audio/webm"):
    # mime → 확장자 매핑
    ext_map = {
        "audio/webm":  ".webm",
        "audio/mp4":   ".m4a",
        "audio/mpeg":  ".mp3",
        "audio/ogg":   ".ogg",
        "audio/wav":   ".wav",
        "audio/x-wav": ".wav",
        "audio/oga":   ".ogg",
    }
    ext = ext_map.get(mime_type, ".webm")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    file_storage.save(tmp.name)
    return tmp.name

def check_size(file_storage):
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > MAX_AUDIO_MB * 1024 * 1024:
        raise ValueError("audio too large")

def make_audio(reply):
    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=reply,
        response_format="mp3",
        speed=1.2, 
        instructions="Speak in a warm, balanced, and gender-neutral voice. Your tone should be inviting and charismatic, striking a perfect balance between a professional concierge and a close friend. Maintain a natural, rhythmic pace with slight pauses to sound thoughtful. Avoid any mechanical stiffness; instead, use a soft, melodic intonation that conveys genuine curiosity and empathy. Your goal is to make the user feel heard, valued, and comfortable sharing their story."
    )
    return base64.b64encode(tts.read()).decode()

# =========================
# TURN endpoint
# =========================

@voice_bp.route("/turn", methods=["POST"])
def voice_turn():

    if "audio" not in request.files:
        return jsonify(error="missing audio"), 400

    # 🔥 프론트에서 전달된 옵션 배열
    import json
    opts_raw   = request.form.get("options")
    opts       = json.loads(opts_raw) if opts_raw else []
    qtype = request.form.get("type", "")
    is_range = qtype == "range"

    path = None
    try:
        audio_file = request.files["audio"]
        check_size(audio_file)
        mime_type = (audio_file.content_type or "audio/webm").split(";")[0].strip()
        path = safe_temp_save(audio_file, mime_type)

        with open(path, "rb") as f:
            stt = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                temperature=0
            )

        transcript = (stt.text or "").strip()

        if not transcript:
            return jsonify(transcript="")
        
        clean = transcript.strip()

        # 최소 한 글자 이상의 알파벳/숫자 포함 필요
        if not re.search(r"[a-zA-Z0-9]", clean):
            return jsonify(transcript="")

        # 너무 짧은 단일 문자 noise 제거 (예: "uh","h")
        if len(clean) <= 1:
            return jsonify(transcript="")

        # -------- GPT mapping --------
        if is_range:
            # age range 자유응답 파싱
            system = """
            You are COCO, a warm dating app companion.
            IMPORTANT: Always reply in English only, regardless of what language the user speaks.
            The user was asked what age range they're looking for.

            Return JSON only:
            {"min": <int>, "max": <int>, "reply": "<one short sentence echoing their range. e.g. 'someone between 25 and 35 — love that.' under 12 words.>"}

            User may speak any language. Translate and extract numbers.
            Examples:
            "20 to 35" → 20, 35
            "mid twenties to early forties" → 24, 42
            "i don't mind, anyone" → 20, 60
            If unclear, default to min:20, max:60.
            """
            gpt = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": transcript}
                ]
            )
            parsed = json.loads(gpt.choices[0].message.content)
            mn     = int(parsed.get("min", 20))
            mx     = int(parsed.get("max", 60))
            reply  = parsed.get("reply", "got it")

            # min > max 방어
            if mn > mx:
                mn, mx = mx, mn

            audio_b64 = make_audio(reply) if reply else None

            return jsonify(
                transcript=transcript,
                min=mn,
                max=mx,
                reply=reply,
                audio=audio_b64
            )

        else:
            base_gender = request.form.get("base_gender", "")
            is_gender_q = set(opts) == {"woman", "man", "non-binary"}
            is_correction = request.form.get("is_correction") == "true"
            prev_answer = request.form.get("prev_answer", "")

            if is_gender_q:
                all_details = [
                    "cis woman", "trans woman", "intersex woman", "transfeminine", "woman and non-binary",
                    "cis man", "trans man", "intersex man", "transmasculine", "man and non-binary",
                    "agender", "bigender", "genderfluid", "genderqueer", "gender nonconforming",
                    "gender questioning", "gender variant", "intersex", "neutrois", "non-binary man",
                    "non-binary woman", "pangender", "polygender", "transgender", "two-spirit"
                ]
                system = f"""
                You are COCO, a warm and emotionally intelligent dating app companion.
                IMPORTANT: Always reply in English only.
                The user was asked: "how do you identify your gender?"
                {"They previously said: " + prev_answer + ". They may be correcting themselves." if is_correction else ""}

                Your job is to understand what they mean — across ANY language, phrasing, or identity term —
                and map it to the closest fit using your judgment.

                BASE gender must be one of: {opts}
                Ask yourself: does this person relate more to woman, man, or neither/outside the binary?
                Be generous and thoughtful. Someone who says "queer", "femme", "they/them", "non-binair",
                "I'm more on the feminine side", "androgynous" — reason about what fits best.
                Only return null if you genuinely cannot infer even a rough direction.

                DETAIL: pick the single closest match from this list, or null if unknown:
                {all_details}

                Return JSON only:
                {{
                "mapped": "<one of {opts} or null>",
                "gender_detail": "<closest match from detail list or null>",
                "reply": "<one warm sentence in English. if mapped is set, echo it and move on — e.g. 'non-binary — love that.' if mapped is null, ask only about base gender.>"
                }}

                Rules:
                - reply MUST be one sentence, under 15 words, English only
                - if mapped is set, NEVER ask a follow-up — just confirm and move on
                - if mapped is null, ask ONLY: woman, man, or non-binary — nothing else
                - if correction, acknowledge naturally: 'oh, trans woman — got it.'
                - NEVER list options robotically
                - NEVER suggest skipping
                - if user says just "woman", "man", "non-binary" with NO detail qualifier, set gender_detail to null — NEVER assume cis or any specific identity
                - only set gender_detail when the user explicitly mentions it
                """
            else:
                question_id = request.form.get("question_id", "")
                
                system = f"""
                You are COCO, a warm and emotionally intelligent dating app companion.
                IMPORTANT: Always reply in English only, regardless of what language the user speaks.
                The user was asked a question with these options: {opts}
                The specific question being asked is about: {question_id}
                {f"The user's base gender is: {base_gender}. Use this context to pick the correct option." if base_gender else ""}
                {"They previously answered: " + prev_answer + ". They may be correcting themselves." if is_correction else ""}

                Return JSON only:
                {{
                "mapped": "<exact option or null>",
                "reply": "<one short sentence only. if mapped, ONLY echo/confirm — never ask follow-up questions. e.g. 'a drinker — love that.' or 'non-smoker, got it.' if null, redirect gently with a hint toward the options.>"
                }}

                Rules:
                - User may speak any language. Understand and map correctly.
                - mapped MUST be verbatim from the list or null
                - Only map if the answer clearly corresponds to one of the options
                - Do NOT map if the transcript is noise, filler words, or completely unrelated
                - If null, reply naturally hints at the options without reading them aloud like a list. Weave 1-2 options into a conversational sentence. e.g. for attraction: 'no worries — do you lean toward men, women, or does it depend on the person?' e.g. for smoking: 'just checking — would you say yes or no to smoking?' Always reference what they actually said if possible.
                - reply MUST be one sentence, under 15 words
                - if user deflects, redirect with a hint toward the actual options — never say 'yes or no' if the options are not yes/no
                - NEVER suggest skipping or moving on
                - if mapped, reply is a confirmation only — NEVER ask how often, how much, or any follow-up
                - reply must reference the actual question topic — NEVER confuse marijuana with smoking
                - reply must be about {question_id} — never reference a different topic
                - Map based on intent: "sometimes", "occasionally", "yeah" → yes; "nah", "not really", "i quit" → no
                - Only return null if genuinely unclear or completely off-topic
                """

            gpt = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": transcript}
                ]
            )
            parsed = json.loads(gpt.choices[0].message.content)
            mapped = parsed.get("mapped")
            if isinstance(mapped, str):
                mapped_norm = mapped.strip().lower()
                opts_norm = {o.lower(): o for o in opts}
                mapped = opts_norm.get(mapped_norm)
            else:
                mapped = None

            gender_detail = parsed.get("gender_detail")
            if isinstance(gender_detail, str):
                gender_detail = gender_detail.strip().lower()
            if is_gender_q:
                valid_gender_details = {
                    "cis woman",
                    "trans woman",
                    "intersex woman",
                    "transfeminine",
                    "woman and non-binary",
                    "cis man",
                    "trans man",
                    "intersex man",
                    "transmasculine",
                    "man and non-binary",
                    "agender",
                    "bigender",
                    "genderfluid",
                    "genderqueer",
                    "gender nonconforming",
                    "gender questioning",
                    "gender variant",
                    "intersex",
                    "neutrois",
                    "non-binary man",
                    "non-binary woman",
                    "pangender",
                    "polygender",
                    "transgender",
                    "two-spirit",
                }
                if gender_detail not in valid_gender_details:
                    gender_detail = None
            reply = parsed.get("reply", "got it")

            audio_b64 = make_audio(reply) if reply else None

            return jsonify(
                transcript=transcript,
                mapped=mapped,
                gender_detail=gender_detail,
                reply=reply,
                audio=audio_b64
            )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error="voice processing failed"), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# =========================
# TTS endpoint
# =========================

@voice_bp.route("/tts", methods=["POST"])
def voice_tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="missing text"), 400

    try:
        audio_b64 = make_audio(text)
        return jsonify(audio=audio_b64)
    except Exception as e:
        print("voice_tts error:", e)
        return jsonify(error="tts failed"), 500


# =========================
# BIO endpoint
# =========================

@voice_bp.route("/bio", methods=["POST"])
def voice_bio():
    data = request.json or {}
    transcripts = data.get("transcripts", [])

    if not transcripts:
        return jsonify(bio="")

    try:

        text = "\n".join(
            f'Q:{t.get("question")} -> "{t.get("said")}"'
            for t in transcripts
        )

        system = """
        You are writing a one-line dating profile bio based on how someone spoke during onboarding.

        Your job is NOT to summarize their answers.
        Your job is to capture the KIND OF PERSON they seem to be — 
        inferred from the attitude, tone, and values behind what they said.

        Rules:
        - exactly one sentence, lowercase, first person
        - read between the lines: what does their wording reveal about their personality?
        e.g. "i don't smoke, but i'd never make that anyone else's problem" 
        → they have standards but aren't judgmental
        - if they said something culturally specific, that's character too
        - do NOT mention specific habits, substances, or lifestyle facts directly
        - do NOT mention the app or finding a match
        - if there is genuinely nothing to infer beyond bare yes/no answers with no elaboration, 
        return an empty string — nothing else, no explanation

        Translate non-English input naturally. Write as if they wrote it themselves.
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":text}
            ]
        )

        bio = resp.choices[0].message.content.strip().strip('"')
        
        if bio.lower() in ("(empty string)", "empty string", "none", "null"):
            bio = ""

        return jsonify(bio=bio)

    except Exception as e:
        print("bio error:", e)
        return jsonify(bio="")

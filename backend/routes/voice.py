from flask import Blueprint, request, jsonify
import base64, tempfile, os
import mimetypes

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
        "audio/webm": ".webm",
        "audio/mp4":  ".mp4",
        "audio/mpeg": ".mp3",
        "audio/ogg":  ".ogg",
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

def make_audio(reply, voice_name):
    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice_name,
        input=reply,
        response_format="mp3",
        speed=1.15, 
        instructions="Speak in clear, natural English with a gender-neutral tone."
    )
    return base64.b64encode(tts.read()).decode()

# =========================
# TURN endpoint
# =========================

COCO_VOICE = "alloy"

@voice_bp.route("/turn", methods=["POST"])
def voice_turn():

    if "audio" not in request.files:
        return jsonify(error="missing audio"), 400

    # 🔥 프론트에서 전달된 옵션 배열
    import json
    opts_raw   = request.form.get("options")
    opts       = json.loads(opts_raw) if opts_raw else []
    is_range   = not opts

    try:
        audio_file = request.files["audio"]
        check_size(audio_file)
        mime_type = audio_file.content_type or "audio/webm"
        path = safe_temp_save(audio_file, mime_type)

        # -------- STT --------
        with open(path, "rb") as f:
            stt = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt="This is a dating app onboarding. The user may speak any language."
            )

        transcript = (stt.text or "").strip()
        os.remove(path)

        if not transcript:
            return jsonify(transcript="")

        # -------- GPT mapping --------
        if is_range:
            # age range 자유응답 파싱
            system = """
            You are COCO, a warm dating app companion.
            The user was asked what age range they're looking for.

            Return JSON only:
            {"min": <int>, "max": <int>, "reply": "<1-2 sentences naturally echoing their preference back. e.g. 'someone between 25 and 35 — a solid range for something real.' or 'you're open to anyone from 20 to 45, i like that openness in you.'>"}

            User may speak any language. Translate and extract numbers.
            Examples:
            "20 to 35" → 20, 35
            "mid twenties to early forties" → 24, 42
            "i don't mind, anyone" → 18, 65
            If unclear, default to min:20, max:40.
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
            mx     = int(parsed.get("max", 40))
            reply  = parsed.get("reply", "got it")

            # min > max 방어
            if mn > mx:
                mn, mx = mx, mn

            audio_b64 = make_audio(reply, COCO_VOICE)

            return jsonify(
                transcript=transcript,
                min=mn,
                max=mx,
                reply=reply,
                audio=audio_b64
            )

        else:
            is_gender_q = set(opts) == {"woman", "man", "non-binary"}
            is_correction = request.form.get("is_correction") == "true"
            prev_answer = request.form.get("prev_answer", "")

            if is_gender_q:
                system = f"""
                You are COCO, a warm and emotionally intelligent dating app companion.
                The user was asked: "how do you identify your gender?"
                {"They previously said: " + prev_answer + ". They may be correcting themselves." if is_correction else ""}

                Map their answer to ONE of: {opts}

                Also check if they mentioned a more specific identity.
                Gender detail options:
                - woman → ["cis woman", "trans woman", "intersex woman", "transfeminine", "woman and non-binary"]
                - man → ["cis man", "trans man", "intersex man", "transmasculine", "man and non-binary"]
                - non-binary → ["agender", "bigender", "genderfluid", "genderqueer", "gender nonconforming"]

                Return JSON only:
                {{
                "mapped": "<exact option or null>",
                "gender_detail": "<exact detail match or null>",
                "reply": "<1-2 sentences: warmly echo what they said using their own words. feel human, not robotic. e.g. 'a trans woman — thank you for sharing that with me.' or 'non-binary and genderfluid, i love that.'>"
                }}

                Rules:
                - User may speak any language. Understand and map correctly.
                - mapped MUST be verbatim from the list or null
                - If correction, acknowledge the correction naturally e.g. 'oh, cis woman — got it, my bad.'
                - If null, reply is a gentle warm re-ask
                """
            else:
                system = f"""
                You are COCO, a warm and emotionally intelligent dating app companion.
                The user was asked a question with these options: {opts}
                {"They previously answered: " + prev_answer + ". They may be correcting themselves." if is_correction else ""}

                Return JSON only:
                {{
                "mapped": "<exact option or null>",
                "reply": "<1-2 sentences that naturally echo their answer in their own spirit. feel like a real person responding, not a bot confirming. e.g. if they said 'yeah socially', reply: 'a social drinker — love that.' if correction: 'oh wait, no worries — got it now.'>"
                }}

                Rules:
                - User may speak any language. Understand and map correctly.
                - mapped MUST be verbatim from the list or null
                - Be generous — map ambiguous answers when possible
                - If null, warm gentle re-ask referencing what they actually said
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
            gender_detail = parsed.get("gender_detail")
            reply = parsed.get("reply", "got it")

            audio_b64 = make_audio(reply, COCO_VOICE)

            return jsonify(
                transcript=transcript,
                mapped=mapped,
                gender_detail=gender_detail,
                reply=reply,
                audio=audio_b64
            )

    except Exception as e:
        print("voice_turn error:", e)
        return jsonify(error="voice processing failed"), 500


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
        audio_b64 = make_audio(text, COCO_VOICE)
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
Create a short natural dating bio.

Rules:
- first person, max 2 sentences, lowercase
- if the user's words are not in English, translate into natural English
- ONLY include genuinely personal expressions: personality traits, passions,
  feelings, humor, life philosophy, unique habits or quirks
- EXCLUDE anything that is already captured as structured data:
  gender identity, gender detail, drinking habits, smoking, marijuana use,
  sexual orientation, age preferences
- If the person's words reveal nothing personal BEYOND those structured fields,
  return exactly: (empty string)
- Do NOT mention the app or matchmaking context
- Write naturally, as if the person wrote it themselves
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

        return jsonify(bio=bio)

    except Exception as e:
        print("bio error:", e)
        return jsonify(bio="")

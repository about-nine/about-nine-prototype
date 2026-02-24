from flask import Blueprint, request, jsonify
import base64, tempfile, os

from openai import OpenAI
client = OpenAI()

voice_bp = Blueprint("voice", __name__, url_prefix="/api/voice")

MAX_AUDIO_MB = 6


# =========================
# helpers
# =========================

def safe_temp_save(file_storage):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
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
        instructions="Speak in clear, natural English with a neutral accent."
    )
    return base64.b64encode(tts.read()).decode()

# =========================
# TURN endpoint
# =========================

@voice_bp.route("/turn", methods=["POST"])
def voice_turn():

    if "audio" not in request.files:
        return jsonify(error="missing audio"), 400

    voice_name = request.form.get("voice", "alloy")

    # 🔥 프론트에서 전달된 옵션 배열
    import json
    opts_raw   = request.form.get("options")
    opts       = json.loads(opts_raw) if opts_raw else []
    is_range   = not opts

    try:
        audio_file = request.files["audio"]
        check_size(audio_file)
        path = safe_temp_save(audio_file)

        # -------- STT --------
        with open(path, "rb") as f:
            stt = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        transcript = (stt.text or "").strip()
        os.remove(path)

        if not transcript:
            return jsonify(transcript="")

        # -------- GPT mapping --------
        if is_range:
            # age range 자유응답 파싱
            system = """
Parse the user's spoken age range preference into min and max integers.
The user may speak any language. Translate/understand it before extracting numbers.
Return JSON only: {"min": <int>, "max": <int>, "reply": "<short warm acknowledgement under 10 words>"}
Examples:
  "20 to 35" → min:20, max:35
  "mid twenties to early forties" → min:24, max:42
  "i don't mind, anyone" → min:18, max:65
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

            audio_b64 = make_audio(reply, voice_name)

            return jsonify(
                transcript=transcript,
                min=mn,
                max=mx,
                reply=reply,
                audio=audio_b64
            )

        else:
            # 일반 ask — opts 중 하나로 매핑
            system = f"""
You are COCO onboarding assistant.

Map the user speech to EXACTLY ONE option from this list:
{opts}

Return JSON only:
{{
  "mapped": "<exact option or null>",
  "reply":  "<short warm acknowledgement under 12 words>"
}}

Rules:
- The user may speak any language. Translate/understand it, then map to the list.
- mapped MUST be verbatim from the list or null
- be generous — ambiguous answers almost always map to something
- if mapped is null (truly off-topic), reply must be a gentle re-ask,
  e.g. "that's interesting — but could you tell me more about your gender?"
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
            mapped = parsed.get("mapped")
            reply  = parsed.get("reply", "got it")

            audio_b64 = make_audio(reply, voice_name)

            return jsonify(
                transcript=transcript,
                mapped=mapped,
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

    voice_name = data.get("voice", "marin")

    try:
        audio_b64 = make_audio(text, voice_name)
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

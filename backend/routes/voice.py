from flask import Blueprint, request, jsonify
import base64, tempfile, os
import traceback

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

def make_audio(reply, voice_name):
    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice_name,
        input=reply,
        response_format="mp3",
        speed=1.2, 
        instructions="Speak in a warm, balanced, and gender-neutral voice. Your tone should be inviting and charismatic, striking a perfect balance between a professional concierge and a close friend. Maintain a natural, rhythmic pace with slight pauses to sound thoughtful. Avoid any mechanical stiffness; instead, use a soft, melodic intonation that conveys genuine curiosity and empathy. Your goal is to make the user feel heard, valued, and comfortable sharing their story."
    )
    return base64.b64encode(tts.read()).decode()

# =========================
# TURN endpoint
# =========================

COCO_VOICE = "echo"

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
        mime_type = (audio_file.content_type or "audio/webm").split(";")[0].strip()
        path = safe_temp_save(audio_file, mime_type)

        # -------- STT --------
        with open(path, "rb") as f:
            question_ctx = request.form.get("question_ctx", "")
            prompt = f"This is a dating app onboarding. The user may speak any language. The question asked was: {question_ctx}" if question_ctx else "This is a dating app onboarding. The user may speak any language."

            stt = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt
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
                - non-binary → ["agender", "bigender", "genderfluid", "genderqueer", "gender nonconforming", "gender questioning", "gender variant", "intersex", "neutrois", "non-binary man", "non-binary woman", "pangender", "polygender", "transgender", "two-spirit"]

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
                - If null, reply naturally hints at options without listing them robotically. e.g. 'totally okay — are you more on the cis side, trans, or somewhere in between?' or 'no worries — do you identify more as a woman, man, or somewhere outside that?'
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
                - If null, reply naturally hints at the options without reading them aloud like a list. Weave 1-2 options into a conversational sentence. e.g. for attraction: 'no worries — do you lean toward men, women, or does it depend on the person?' e.g. for smoking: 'just checking — would you say yes or no to smoking?' Always reference what they actually said if possible.
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
                mapped = mapped.strip().lower()
            if mapped not in opts:
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

            audio_b64 = make_audio(reply, COCO_VOICE)

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

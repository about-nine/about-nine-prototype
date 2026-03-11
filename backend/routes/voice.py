# routes/voice.py
from flask import Blueprint, request, jsonify
import base64, json, re, traceback
from openai import OpenAI

client = OpenAI()
voice_bp = Blueprint("voice", __name__, url_prefix="/api/voice")

REQUIRED_FIELDS = ["gender", "gender_detail", "sexual_orientation", "age_preference", "drink", "smoke", "marijuana"]

GENDER_DETAIL_OPTIONS = {
    "woman": ["cis woman", "trans woman", "intersex woman", "transfeminine", "woman and non-binary"],
    "man": ["cis man", "trans man", "intersex man", "transmasculine", "man and non-binary"],
    "non-binary": [
        "agender", "bigender", "genderfluid", "genderqueer", "gender nonconforming",
        "gender questioning", "gender variant", "intersex", "neutrois",
        "non-binary man", "non-binary woman", "pangender", "polygender",
        "transgender", "two-spirit",
    ],
}

SEXUAL_ORIENTATION_OPTIONS = [
    "men", "women", "men and women",
    "men and non-binary people", "women and non-binary people",
    "all types of genders",
]

_STT_CORRECTIONS = [
    # --- gender_detail ---
    # "cis" sounds like "this", "sis", "siz", "six", "chris", "sus", "susse", "sussman"
    (r"\b(?:this|sis|siz|six|chris)\s+(woman|man)\b", r"cis \1"),
    (r"\b(?:susse?man|sus\s+man|suss\s+man)\b", "cis man"),
    (r"\b(?:suswoman|sus\s+woman)\b", "cis woman"),
    (r"\bcis[-\s]?(woman|man)\b", r"cis \1"),
    # "trans" variations
    (r"\btrans[-\s]?(woman|man|gender|masculine|feminine|feminin)\b", r"trans\1"),
    (r"\btransgender\b", "transgender"),
    # "non-binary" spacing/hyphen variations
    (r"\bnon[-\s]binary\b", "non-binary"),
    (r"\bnone[-\s]?binary\b", "non-binary"),
    # "intersex" spacing
    (r"\binter[-\s]sex\b", "intersex"),
    # "genderfluid" spacing
    (r"\bgender[-\s]fluid\b", "genderfluid"),
    (r"\bgender[-\s]queer\b", "genderqueer"),
    (r"\bgender[-\s]nonconforming\b", "gender nonconforming"),
    (r"\bnon[-\s]binary[-\s](woman|man)\b", r"non-binary \1"),
    # "two-spirit" spacing
    (r"\btwo[-\s]spirit\b", "two-spirit"),
    # --- sexual_orientation ---
    # singular → plural
    (r"\bman and woman\b", "men and women"),
    (r"\bman and women\b", "men and women"),
    (r"\bmen and woman\b", "men and women"),
    (r"\bmen and non[-\s]?binary(?:\s+people)?\b", "men and non-binary people"),
    (r"\bwomen and non[-\s]?binary(?:\s+people)?\b", "women and non-binary people"),
    # --- marijuana synonyms ---
    (r"\b(?:weed|cannabis|pot|ganja|herb|mary\s*jane|marry\s*(?:wanna|wana|juana?))\b", "marijuana"),
    # --- drink/smoke normalizations ---
    (r"\b(?:yeah|yep|yup|sure|absolutely|definitely)\b", "yes"),
    (r"\b(?:nope|nah|never)\b", "no"),
]
_STT_CORRECTION_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in _STT_CORRECTIONS]

def apply_stt_corrections(text: str) -> str:
    for pattern, replacement in _STT_CORRECTION_COMPILED:
        text = pattern.sub(replacement, text)
    return text


_HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"https?://",                                   # URLs like otter.ai
    ]
]

def _is_hallucination(text: str) -> bool:
    if not text:
        return False
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def normalize_gender_detail(value: str, gender: str) -> str | None:
    """Map free-form LLM output to one of the canonical gender_detail values."""
    v = value.strip().lower()
    allowed = GENDER_DETAIL_OPTIONS.get(gender, [])
    if v in allowed:
        return v
    # common variation mappings
    _map = {
        "cisgender woman": "cis woman", "cis-woman": "cis woman",
        "cisgender man": "cis man", "cis-man": "cis man",
        "transgender woman": "trans woman", "trans-woman": "trans woman",
        "transgender man": "trans man", "trans-man": "trans man",
        "transgender": "transgender",
        "intersex woman": "intersex woman", "intersex man": "intersex man",
        "gender fluid": "genderfluid", "gender-fluid": "genderfluid",
        "gender queer": "genderqueer", "gender-queer": "genderqueer",
        "gender nonconforming": "gender nonconforming",
        "gender non-conforming": "gender nonconforming",
        "gender non conforming": "gender nonconforming",
        "two spirit": "two-spirit",
        "non binary man": "non-binary man", "non binary woman": "non-binary woman",
    }
    mapped = _map.get(v)
    if mapped and mapped in allowed:
        return mapped
    return None


def normalize_sexual_orientation(value: str) -> str | None:
    """Map free-form LLM output to one of the canonical sexual_orientation values."""
    v = value.strip().lower()
    if v in SEXUAL_ORIENTATION_OPTIONS:
        return v
    # broad mappings
    if v in ("everyone", "all", "all genders", "anyone", "all people", "all gender", "everybody", "any gender"):
        return "all types of genders"
    if "non-binary" in v and "men" in v and "women" in v:
        return "all types of genders"
    if "men and non-binary" in v:
        return "men and non-binary people"
    if "women and non-binary" in v:
        return "women and non-binary people"
    if "men and women" in v or "women and men" in v or v in ("bisexual", "bi"):
        return "men and women"
    if v in ("pansexual", "pan", "queer", "fluid", "omnisexual"):
        return "all types of genders"
    if v == "men" or v == "man" or v == "male" or v == "males" or v == "guys":
        return "men"
    if v == "women" or v == "woman" or v == "female" or v == "females":
        return "women"
    if "non-binary" in v:
        return "all types of genders"
    return None


# =========================
# helpers
# =========================

_NUDGE_OPTIONS = {
    "gender_detail": lambda collected: GENDER_DETAIL_OPTIONS.get(collected.get("gender", ""), []),
    "sexual_orientation": lambda _: SEXUAL_ORIENTATION_OPTIONS,
    "age_preference": lambda _: ["e.g. 25 to 35", "e.g. 28 to 40"],
    "drink": lambda _: ["yes", "no"],
    "smoke": lambda _: ["yes", "no"],
    "marijuana": lambda _: ["yes", "no"],
}

def _nudge_instruction(field: str, collected: dict, round_num: int = 1) -> str:
    get_opts = _NUDGE_OPTIONS.get(field)
    if not get_opts:
        return ""
    opts = get_opts(collected)
    sample = opts[:3] if len(opts) > 3 else opts
    if round_num >= 2:
        return (
            f"CRITICAL: The user still hasn't answered '{field}'. "
            f"Ask ONLY about '{field}' and include 2-3 examples inline: {sample}. "
            f"One sentence. Use noticeably different wording from before."
        )
    return (
        f"CRITICAL: The user hasn't answered '{field}'. "
        f"Rephrase the question in a new way. One sentence."
    )


def make_audio(reply):
    tts = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="shimmer",
        input=reply,
        response_format="mp3",
        speed=1.0,
        instructions="Speak in a soft, gentle, whispering tone. Keep your voice calm, quiet, and intimate, like you're telling a secret."
    )
    return base64.b64encode(tts.read()).decode()


# =========================
# TURN endpoint
# =========================

@voice_bp.route("/turn", methods=["POST"])
def voice_turn():
    data = request.get_json(silent=True) or {}
    history = data.get("history") or []
    collected = data.get("collected") or {}
    first_name = (data.get("firstName") or "").strip()
    deferred_fields = data.get("deferred_fields") or []
    if not isinstance(deferred_fields, list):
        deferred_fields = []
    deferred_fields = [f for f in deferred_fields if f in REQUIRED_FIELDS]
    nudge_round_raw = data.get("nudge_round") or 1
    try:
        nudge_round = int(nudge_round_raw)
    except (TypeError, ValueError):
        nudge_round = 1

    missing_required = [k for k in REQUIRED_FIELDS if not collected.get(k)]
    is_initial = not any(m.get("role") == "user" for m in history)
    all_done = not missing_required
    nudge_field = data.get("nudge_field") or None
    active_missing = [k for k in missing_required if k not in deferred_fields]
    target_field = None
    if nudge_field and nudge_field in missing_required:
        target_field = nudge_field
    elif active_missing:
        target_field = active_missing[0]
    elif missing_required:
        target_field = missing_required[0]

    target_instruction = ""
    if target_field == "gender_detail":
        if collected.get("gender"):
            target_instruction = (
                "gender_detail is still needed — their gender is already known "
                f"({collected['gender']}). Ask specifically about gender_detail (this is different from gender). "
                f"Give 2-3 examples: {GENDER_DETAIL_OPTIONS.get(collected['gender'], [])}."
            )
        else:
            target_instruction = (
                "gender_detail is still needed — once gender is known, immediately ask for gender_detail "
                "with examples like: cis man, trans man, or something else."
            )
    elif target_field == "gender":
        target_instruction = "Ask how they identify in terms of gender."
    elif target_field == "sexual_orientation":
        target_instruction = "Ask who they are attracted to."
    elif target_field == "age_preference":
        target_instruction = "Ask for the age range they are looking for."
    elif target_field == "drink":
        target_instruction = "Ask if they drink alcohol."
    elif target_field == "smoke":
        target_instruction = "Ask if they smoke cigarettes."
    elif target_field == "marijuana":
        target_instruction = "Ask if they use marijuana."

    system = f"""You are COCO, a warm and emotionally intelligent companion for a dating app.
You're having a natural onboarding conversation with {first_name or "someone new"}.
Your goal: learn about them through genuine conversation — not a questionnaire.

You need to collect through conversation:

REQUIRED (must all be gathered before ending):
- gender: "woman", "man", or "non-binary"
- gender_detail: more specific identity{f" — allowed values for a {collected['gender']}: {GENDER_DETAIL_OPTIONS.get(collected['gender'], [])}" if collected.get("gender") else " (ask after learning their gender)"}
- sexual_orientation: who they're attracted to
- age_preference: age range they're looking for (min and max as integers)
- drink: "yes" or "no" (alcohol)
- smoke: "yes" or "no" (cigarettes)
- marijuana: "yes" or "no"

Already collected: {json.dumps(collected)}
Still needed (REQUIRED): {missing_required}

{"This is the very start of the conversation. In one sentence: greet them by name (" + (first_name or "their name") + "), introduce yourself as COCO, and ask how they identify in terms of gender." if is_initial else ""}
{"All required info is collected — wrap up the conversation warmly and naturally." if all_done else ""}
{target_instruction if target_instruction else ""}
{_nudge_instruction(nudge_field, collected, nudge_round) if nudge_field else ""}

Conversation rules:
- Be warm and human, like a friend — NOT a form or survey
- Ask only one thing at a time
- If they mention optional info naturally, note it
- If they're unclear, ask gently without repeating a list of options robotically
- If they change their mind, update accordingly
- Never suggest skipping or say things like "moving on" or "got it" robotically
- Never ask how someone feels about their identity — accept what they say and move on
- If you need to ask about something the user already touched on but wasn't clear enough to collect, reference what they said rather than asking from scratch (e.g. "You mentioned you drink socially — so you do drink, yeah?" instead of "Do you drink alcohol?")
- If the same field keeps failing across multiple turns, rephrase the question completely — describe it differently or offer a brief list of options rather than asking the same way again
- Phrase questions to naturally elicit a sentence rather than a single word (e.g. "how do you identify?" rather than "man, woman, or non-binary?") — longer responses are easier to understand
- Speech recognition may mishear certain words. When extracting fields, consider: "this/sis woman" likely means "cis woman", "non binary" means "non-binary", "weed/cannabis" means marijuana, "amen" when discussing gender likely means "a man"
- If the user says something off-topic (greetings, farewells, unrelated comments), do not engage with it — briefly redirect back to the current question
- If the user expresses frustration, confusion, or says things like "I can't do this", "I'm lost", "what are you doing", swear words, or any emotional distress — do NOT acknowledge or engage with the emotional content. Treat it exactly like off-topic input: redirect back to the current question in one sentence. Never offer to take a break, pause, or talk about something else.
- ONE sentence only. Maximum 15 words. Never two sentences. Never a follow-up clause.
- Never use lists, options, or multiple questions in one turn
- Always reply in English

Return JSON only:
{{
  "reply": "<your conversational response>",
  "collected": {{<new fields you can extract from THIS turn only — empty dict if none>}},
  "is_complete": <true only when ALL required fields are now collected, including this turn>
}}

Field formats for "collected":
- gender: "woman" | "man" | "non-binary"
- gender_detail: one of the allowed values listed above — omit if not mentioned
- drink / smoke / marijuana: "yes" | "no"
- sexual_orientation: must be exactly one of: "men" | "women" | "men and women" | "men and non-binary people" | "women and non-binary people" | "all types of genders"
- age_preference: {{"min": <int>, "max": <int>}}

Age preference guidance:
- If the user gives vague or decade-style ranges (e.g., "mid twenties", "late 30s", "20대 중반", "not too young/old", "상관없어"),
  set age_preference to {{"min": 20, "max": 60}}.

Only extract what was clearly said in this turn. Do not re-extract already-collected fields."""

    messages = [{"role": "system", "content": system}]
    for msg in history[-20:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        gpt = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,

            response_format={"type": "json_object"},
            messages=messages,
        )
        parsed = json.loads(gpt.choices[0].message.content)
        reply = parsed.get("reply", "")
        new_collected = parsed.get("collected") or {}
        is_complete = bool(parsed.get("is_complete", False))

        # merge new fields into collected
        merged = dict(collected)
        for k, v in new_collected.items():
            if v is not None and v != "":
                merged[k] = v

        # normalize sexual_orientation to canonical value
        so = merged.get("sexual_orientation")
        if so is not None:
            normalized_so = normalize_sexual_orientation(so)
            if normalized_so:
                merged["sexual_orientation"] = normalized_so
            else:
                merged.pop("sexual_orientation", None)

        # normalize gender_detail to canonical value
        gd = merged.get("gender_detail")
        if gd is not None:
            gender = merged.get("gender", "")
            normalized_gd = normalize_gender_detail(gd, gender)
            if normalized_gd:
                merged["gender_detail"] = normalized_gd
            else:
                merged.pop("gender_detail", None)

        # normalize age_preference: accept {min, max} dict or "25-40" / "25 to 40" strings
        ap = merged.get("age_preference")
        if ap is not None:
            if isinstance(ap, dict):
                mn = ap.get("min")
                mx = ap.get("max")
                if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
                    mn, mx = int(mn), int(mx)
                    if mn > mx:
                        mn, mx = mx, mn
                    merged["age_preference"] = {"min": mn, "max": mx}
                else:
                    merged.pop("age_preference", None)
        elif isinstance(ap, str):
            m = re.search(r"(\d+)\D+(\d+)", ap)
            if m:
                mn, mx = int(m.group(1)), int(m.group(2))
                if mn > mx:
                    mn, mx = mx, mn
                merged["age_preference"] = {"min": mn, "max": mx}
            else:
                merged.pop("age_preference", None)

        # server-side guard: is_complete only if all required fields present with valid values
        all_present = all(merged.get(req) for req in REQUIRED_FIELDS)
        if is_complete and not all_present:
            is_complete = False
        # force-complete if LLM forgot to set it but everything is collected
        if not is_complete and all_present:
            is_complete = True

        audio_b64 = make_audio(reply) if reply else None

        return jsonify(
            reply=reply,
            audio=audio_b64,
            collected=merged,
            is_complete=is_complete,
            target_field=target_field,
        )

    except Exception:
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
        return jsonify(audio=make_audio(text))
    except Exception as e:
        print("voice_tts error:", e)
        return jsonify(error="tts failed"), 500


# =========================
# STT endpoint
# =========================

@voice_bp.route("/stt", methods=["POST"])
def voice_stt():
    audio = request.files.get("audio")
    if not audio:
        return jsonify(error="no audio"), 400
    language = request.form.get("language") or None
    field_hint = request.form.get("field_hint") or None

    _FIELD_PROMPTS = {
        "gender": "woman man non-binary",
        "gender_detail": "cis man cis woman trans man trans woman non-binary intersex genderfluid genderqueer",
        "sexual_orientation": "men women men and women all types of genders non-binary people",
        "age_preference": "twenty thirty forty fifty years old to between",
        "drink": "yes no I drink I don't drink alcohol",
        "smoke": "yes no I smoke I don't smoke cigarettes",
        "marijuana": "yes no I use I don't use marijuana weed cannabis",
    }
    whisper_prompt = _FIELD_PROMPTS.get(field_hint) if field_hint else None

    try:
        kwargs = dict(
            model="whisper-1",
            file=(audio.filename, audio.stream, audio.mimetype),
            language=language,
            response_format="verbose_json",
        )
        if whisper_prompt:
            kwargs["prompt"] = whisper_prompt
        transcript = client.audio.transcriptions.create(**kwargs)

        # Filter hallucinations via no_speech_prob
        no_speech_prob = getattr(transcript, "no_speech_prob", None)
        if no_speech_prob is not None and no_speech_prob > 0.6:
            print(f"[STT] hallucination rejected (no_speech_prob={no_speech_prob:.2f})")
            return jsonify(transcript="")

        text = (transcript.text or "").strip()

        # Filter known Whisper hallucination patterns
        if _is_hallucination(text):
            print(f"[STT] hallucination rejected (pattern): {text[:80]}")
            return jsonify(transcript="")

        text = apply_stt_corrections(text)
        # For yes/no fields, "you" is likely a mishearing of "no"
        if field_hint in ("drink", "smoke", "marijuana"):
            if text.strip().lower() in ("you", "you.", "you?"):
                text = "no"
        return jsonify(transcript=text)
    except Exception as e:
        print("STT error:", e)
        return jsonify(error="stt failed"), 500


# =========================
# BIO endpoint
# =========================

@voice_bp.route("/bio", methods=["POST"])
def voice_bio():
    data = request.json or {}
    history = data.get("history") or []

    user_messages = [
        m["content"] for m in history
        if m.get("role") == "user" and m.get("content")
    ]
    if not user_messages:
        return jsonify(bio="")

    try:
        text = "\n".join(f'"{msg}"' for msg in user_messages)

        system = """You are writing a one-line dating profile bio based on how someone spoke during onboarding.

Your job is NOT to summarize their profile answers.
Your job is to capture the KIND OF PERSON they seem to be —
inferred from their attitude, tone, word choice, and values.

NEVER include anything related to:
- gender or gender identity
- sexual orientation or who they're attracted to
- age or age preferences
- drinking, smoking, or marijuana use
These are stored separately and must NOT appear in the bio.

Rules:
- exactly one sentence, lowercase, first person
- only write if there is genuine personality to infer beyond the profile questions
- if the user only gave bare yes/no answers with no personality showing, return an empty string
- do NOT mention the app or finding a match

Translate non-English input naturally. Write as if they wrote it themselves."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        bio = resp.choices[0].message.content.strip().strip('"')
        if bio.lower() in ("(empty string)", "empty string", "none", "null"):
            bio = ""
        return jsonify(bio=bio)

    except Exception as e:
        print("bio error:", e)
        return jsonify(bio="")

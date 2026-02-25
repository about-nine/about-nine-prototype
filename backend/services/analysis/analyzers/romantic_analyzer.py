# services/analysis/analyzers/romantic_analyzer.py - romantic intent analyzer
"""
Romantic Intent Analyzer (romantic_intent) - 다국어/한국어 최적화
==================================================================

■ 현재 코드 기준 특징
────────────────────────────────────────────────────────────────
[1] Seed Examples: 언어별 seed 분리
    - en/ko + es/fr/de/pt 예시 포함
    - 발화 언어에 따라 해당 언어 seed만 임베딩 비교

[2] 언어 감지 기반 Threshold
    - en 0.40, ko 0.35
    - es/fr/pt 0.38, de 0.39, ja/zh 0.36

[3] 한국어 문화 반영 Intensity
    - 카테고리 메타데이터에 intensity/intensity_ko 분리
    - Care(ko=0.72), Intimacy(ko=0.85) 등 한국어 보정값 적용

[4] Keyword Fallback: 언어별 분리
    - 영어: 단어/구 단위 매칭 (EN_KEYWORD_MAP)
    - 한국어: 어간 기반 정규식 (KO_REGEX_MAP)
    - es/fr/de/pt: 키워드 매칭 (LANG_KEYWORD_MAPS)

[5] Mutuality 패턴 보강
    - 영어/한국어 + es/fr/de/pt 패턴 포함

[6] 카테고리 메타데이터 완비
    - Curiosity/Playfulness/Availability 포함
    - 점수 포함 카테고리(ROMANTIC_CATEGORIES)와
      분류용 전체 카테고리(ROMANTIC_ALL_CATEGORIES) 분리

측정 원리:
  1. 발화별 언어 감지 (language_utils)
  2. OpenAI Embedding으로 seed와 코사인 유사도 비교
     (언어별 threshold 적용)
  3. intensity × coverage 기반 최종 0~100점 환산
  4. Fallback: 언어별 키워드/정규식 매칭

의존성: openai (embedding, optional), numpy
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from analyzers.language_utils import detect_conversation_language, detect_utterance_languages

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

EMBEDDING_THRESHOLD_BY_LANG: Dict[str, float] = {
    "ko": 0.35,
    "ja": 0.36,
    "zh": 0.36,
    "en": 0.40,
    "es": 0.38,
    "fr": 0.38,
    "de": 0.39,
    "pt": 0.38,
    "default": 0.40,
}

ROMANTIC_DECISION_THRESHOLD_BY_LANG: Dict[str, float] = {
    "ko": 0.42,
    "ja": 0.44,
    "zh": 0.44,
    "default": 0.45,
}

ROMANTIC_CATEGORIES = {
    "Longing",
    "Adoration",
    "Affection",
    "Care",
    "Intimacy",
    "Physical Response",
    "Future Together",
    "Exclusivity",
}

ROMANTIC_ALL_CATEGORIES = [
    "Longing",
    "Adoration",
    "Affection",
    "Care",
    "Intimacy",
    "Physical Response",
    "Future Together",
    "Exclusivity",
    "Curiosity",
    "Playfulness",
    "Availability",
]

ROMANTIC_SEED_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Longing": {
        "examples_en": [
            "I miss you so much",
            "I can't wait to see you",
            "I wish you were here",
            "I keep thinking about you",
            "I wish I could be with you right now",
        ],
        "examples_ko": [
            "보고 싶어", "보고파", "보고싶다", "보고 싶은데",
            "빨리 만나고 싶어", "언제 봐", "언제 와",
            "자꾸 생각나", "네 생각 나", "계속 생각나",
            "기다리고 있어", "기다려", "빨리 왔으면",
            "생각나더라",
        ],
        "examples_es": [
            "Te extraño mucho",
            "No puedo esperar a verte",
            "Ojalá estuvieras aquí",
            "No dejo de pensar en ti",
            "Quisiera estar contigo ahora mismo",
        ],
        "examples_fr": [
            "Tu me manques tellement",
            "J'ai hâte de te voir",
            "J'aimerais que tu sois ici",
            "Je pense sans arrêt à toi",
            "J'aimerais être avec toi maintenant",
        ],
        "examples_de": [
            "Ich vermisse dich so sehr",
            "Ich kann es kaum erwarten, dich zu sehen",
            "Ich wünschte, du wärst hier",
            "Ich denke ständig an dich",
            "Ich wünschte, ich könnte jetzt bei dir sein",
        ],
        "examples_pt": [
            "Estou com muita saudade de você",
            "Mal posso esperar para te ver",
            "Queria que você estivesse aqui",
            "Não paro de pensar em você",
            "Queria estar com você agora",
        ],
        "intensity": 0.85,
        "intensity_ko": 0.85,
        "directness": 0.80,
    },
    "Adoration": {
        "examples_en": [
            "You're so beautiful",
            "You're amazing",
            "I think you're incredible",
            "You have such a lovely smile",
            "You're so attractive",
        ],
        "examples_ko": [
            "너 정말 예뻐", "너 진짜 멋있다", "너 너무 귀엽다",
            "예쁜데", "잘생겼다", "이쁘다",
            "눈빛이 예뻐", "웃을 때 예뻐", "목소리가 좋아",
            "완전 내 스타일", "너 보면 설레",
        ],
        "examples_es": [
            "Eres hermosa",
            "Eres increíble",
            "Tienes una sonrisa preciosa",
            "Qué guapa eres",
            "Eres muy atractiva",
        ],
        "examples_fr": [
            "Tu es magnifique",
            "Tu es incroyable",
            "Tu as un sourire adorable",
            "Tu es très belle",
            "Tu es splendide",
        ],
        "examples_de": [
            "Du bist wunderschön",
            "Du bist unglaublich",
            "Dein Lächeln ist wunderschön",
            "Du bist so attraktiv",
            "Du bist toll",
        ],
        "examples_pt": [
            "Você é linda",
            "Você é incrível",
            "Seu sorriso é lindo",
            "Você é maravilhosa",
            "Você é muito atraente",
        ],
        "intensity": 0.80,
        "intensity_ko": 0.80,
        "directness": 0.90,
    },
    "Affection": {
        "examples_en": [
            "I love you",
            "I really like you",
            "You mean everything to me",
            "I have feelings for you",
            "I've fallen for you",
        ],
        "examples_ko": [
            "좋아해", "사랑해", "많이 좋아해", "정말 좋아해",
            "엄청 좋아해", "좋아하는 것 같아",
            "좋아하게 된 것 같아", "빠진 것 같아",
            "너 때문에 행복해", "너라서 좋아",
        ],
        "examples_es": [
            "Te quiero",
            "Te amo",
            "Me gustas mucho",
            "Tengo sentimientos por ti",
            "Me he enamorado de ti",
        ],
        "examples_fr": [
            "Je t'aime",
            "Je t'adore",
            "Tu me plais beaucoup",
            "J'ai des sentiments pour toi",
            "Je suis tombé amoureux(se) de toi",
        ],
        "examples_de": [
            "Ich liebe dich",
            "Ich mag dich sehr",
            "Ich habe Gefühle für dich",
            "Ich habe mich in dich verliebt",
            "Du bedeutest mir viel",
        ],
        "examples_pt": [
            "Eu te amo",
            "Eu gosto muito de você",
            "Tenho sentimentos por você",
            "Estou apaixonado(a) por você",
            "Você significa muito para mim",
        ],
        "intensity": 1.00,
        "intensity_ko": 1.00,
        "directness": 1.00,
    },
    "Care": {
        "examples_en": [
            "Have you eaten?",
            "Take care of yourself",
            "I'm worried about you",
            "Get some rest",
            "Don't push yourself too hard",
        ],
        "examples_ko": [
            "밥 먹었어", "밥은 먹었어", "밥 챙겨 먹어",
            "식사 했어", "잘 잤어", "잘 자", "잘 들어가",
            "피곤하겠다", "힘들겠다", "걱정돼",
            "걱정되잖아", "조심해", "건강 챙겨",
            "아프지 마", "많이 힘들었겠다", "고생했어",
            "잘하고 있어",
        ],
        "examples_es": [
            "¿Has comido?",
            "Cuídate",
            "Me preocupas",
            "Descansa",
            "No te esfuerces tanto",
        ],
        "examples_fr": [
            "Tu as mangé ?",
            "Prends soin de toi",
            "Je m'inquiète pour toi",
            "Repose-toi",
            "Ne te surmène pas",
        ],
        "examples_de": [
            "Hast du gegessen?",
            "Pass auf dich auf",
            "Ich mache mir Sorgen um dich",
            "Ruh dich aus",
            "Übertreib's nicht",
        ],
        "examples_pt": [
            "Você já comeu?",
            "Se cuida",
            "Estou preocupado(a) com você",
            "Descansa",
            "Não se esforce tanto",
        ],
        # 한국어 Care는 문화적으로 더 강한 애정 표현
        "intensity": 0.60,
        "intensity_ko": 0.72,
        "directness": 0.50,
    },
    "Intimacy": {
        "examples_en": [
            "My love",
            "Sweetheart",
            "Honey",
            "Babe",
            "Darling",
        ],
        "examples_ko": [
            "자기야", "자기", "내 사랑", "여보",
            "우리 자기", "자기가", "자기 보고 싶어",
            "귀요미", "아가",
        ],
        "examples_es": [
            "Mi amor",
            "Cariño",
            "Mi vida",
            "Bebé",
            "Corazón",
        ],
        "examples_fr": [
            "Mon amour",
            "Chéri",
            "Chérie",
            "Mon cœur",
            "Bébé",
        ],
        "examples_de": [
            "Schatz",
            "Liebling",
            "Süße",
            "Babe",
            "Mein Herz",
        ],
        "examples_pt": [
            "Meu amor",
            "Querido",
            "Querida",
            "Meu bem",
            "Bebê",
        ],
        "intensity": 0.80,
        "intensity_ko": 0.85,
        "directness": 0.90,
    },
    "Physical Response": {
        "examples_en": [
            "My heart is racing",
            "I feel butterflies",
            "You make me nervous",
            "I can feel my heart pounding",
            "I get so flustered around you",
        ],
        "examples_ko": [
            "심장 두근거려", "두근두근", "설레", "설렌다",
            "설레는데", "떨려", "긴장돼",
            "심장이 쿵쾅", "얼굴이 빨개지는 것 같아",
            "나도 모르게 웃게 돼", "심장 뛰어", "두근거리는데",
        ],
        "examples_es": [
            "Se me acelera el corazón",
            "Siento mariposas",
            "Me pongo nervioso(a) contigo",
            "Me late el corazón",
            "Me haces temblar",
        ],
        "examples_fr": [
            "Mon cœur bat vite",
            "J'ai des papillons",
            "Tu me rends nerveux(se)",
            "Mon cœur s'emballe",
            "Je rougis",
        ],
        "examples_de": [
            "Mein Herz rast",
            "Ich habe Schmetterlinge im Bauch",
            "Du machst mich nervös",
            "Mein Herz klopft",
            "Ich werde rot",
        ],
        "examples_pt": [
            "Meu coração acelera",
            "Sinto borboletas",
            "Você me deixa nervoso(a)",
            "Meu coração dispara",
            "Fico vermelho(a)",
        ],
        "intensity": 0.85,
        "intensity_ko": 0.85,
        "directness": 0.70,
    },
    "Future Together": {
        "examples_en": [
            "Let's travel together someday",
            "We should do this again",
            "I want to spend more time with you",
            "Are you free this weekend",
            "We should meet up again",
        ],
        "examples_ko": [
            "같이 여행 가자", "같이 밥 먹자", "다음에 또 만나자",
            "우리 또 보자", "같이 가보고 싶은 데 있어",
            "나중에 같이", "언제 시간 돼",
            "주말에 뭐 해", "다음에는 어디 갈까",
            "또 보고 싶다", "다음에 봐",
            "우리 언제 볼 수 있어",
        ],
        "examples_es": [
            "Viajemos juntos algún día",
            "Deberíamos hacerlo de nuevo",
            "Quiero pasar más tiempo contigo",
            "¿Estás libre este fin de semana?",
            "Nos vemos otra vez",
        ],
        "examples_fr": [
            "On devrait refaire ça",
            "Je veux passer plus de temps avec toi",
            "Tu es libre ce week-end ?",
            "Partons ensemble un jour",
            "On se voit encore",
        ],
        "examples_de": [
            "Lass uns das wiederholen",
            "Ich möchte mehr Zeit mit dir verbringen",
            "Bist du am Wochenende frei?",
            "Lass uns zusammen reisen",
            "Wir sollten uns wiedersehen",
        ],
        "examples_pt": [
            "Vamos viajar juntos algum dia",
            "Vamos fazer isso de novo",
            "Quero passar mais tempo com você",
            "Você está livre no fim de semana?",
            "A gente se vê de novo",
        ],
        "intensity": 0.75,
        "intensity_ko": 0.75,
        "directness": 0.70,
    },
    "Exclusivity": {
        "examples_en": [
            "You're the only one for me",
            "You're special to me",
            "I've never felt this way before",
            "There's no one else like you",
            "You're unlike anyone I've ever met",
        ],
        "examples_ko": [
            "너만 이래", "너만 그래", "너밖에 없어",
            "너한테만 이런 거야", "다른 사람한테는 이러지 않는데",
            "특별해", "특별한 것 같아",
            "처음이야", "처음 느끼는 감정이야", "이런 감정 처음이야",
        ],
        "examples_es": [
            "Eres la única persona para mí",
            "Eres especial para mí",
            "Nunca había sentido esto",
            "No hay nadie como tú",
            "Solo tú",
        ],
        "examples_fr": [
            "Tu es la seule pour moi",
            "Tu es spéciale pour moi",
            "Je n'ai jamais ressenti ça",
            "Il n'y a personne comme toi",
            "Toi et personne d'autre",
        ],
        "examples_de": [
            "Du bist die Einzige für mich",
            "Du bist etwas Besonderes für mich",
            "So habe ich mich noch nie gefühlt",
            "Es gibt niemanden wie dich",
            "Nur du",
        ],
        "examples_pt": [
            "Você é a única para mim",
            "Você é especial para mim",
            "Nunca senti isso antes",
            "Não há ninguém como você",
            "Só você",
        ],
        "intensity": 0.95,
        "intensity_ko": 0.95,
        "directness": 0.85,
    },
    "Curiosity": {
        "examples_en": [
            "Tell me more about that",
            "What do you do for fun",
            "How did you get into that",
            "That's really interesting",
            "I'd love to know more about you",
        ],
        "examples_ko": [
            "더 알고 싶어", "어떻게 됐어", "어떤 사람이야",
            "뭐 좋아해", "평소에 뭐 해", "취미가 뭐야",
            "진짜 궁금해", "흥미롭다", "재밌다 그거",
        ],
        "examples_es": [
            "Cuéntame más",
            "¿Qué te gusta hacer?",
            "¿Cómo empezaste?",
            "Eso es interesante",
            "Quiero saber más de ti",
        ],
        "examples_fr": [
            "Dis-m'en plus",
            "Tu fais quoi pour t'amuser ?",
            "Comment tu t'y es mis(e) ?",
            "C'est intéressant",
            "J'aimerais en savoir plus sur toi",
        ],
        "examples_de": [
            "Erzähl mir mehr",
            "Was machst du gern in deiner Freizeit?",
            "Wie bist du dazu gekommen?",
            "Das ist interessant",
            "Ich möchte mehr über dich wissen",
        ],
        "examples_pt": [
            "Me conta mais",
            "O que você gosta de fazer?",
            "Como você começou?",
            "Isso é interessante",
            "Quero saber mais sobre você",
        ],
        "intensity": 0.45,
        "intensity_ko": 0.45,
        "directness": 0.40,
    },
    "Playfulness": {
        "examples_en": [
            "Haha you're so funny",
            "Stop you're making me laugh",
            "Are you always this charming",
            "You're such a tease",
        ],
        "examples_ko": [
            "ㅋㅋㅋ 너 웃겨", "하하 재밌다", "놀리는 거야",
            "장난이야", "귀엽게 구네", "항상 이래",
        ],
        "examples_es": [
            "Jaja eres muy gracioso(a)",
            "Me haces reír",
            "Eres un(a) bromista",
            "Qué divertido(a) eres",
        ],
        "examples_fr": [
            "Haha tu es drôle",
            "Tu me fais rire",
            "Tu es taquin(e)",
            "Tu es vraiment amusant(e)",
        ],
        "examples_de": [
            "Haha du bist lustig",
            "Du bringst mich zum Lachen",
            "Du bist so witzig",
            "Du neckst mich",
        ],
        "examples_pt": [
            "Haha você é engraçado(a)",
            "Você me faz rir",
            "Você é brincalhão(a)",
            "Você é muito divertido(a)",
        ],
        "intensity": 0.40,
        "intensity_ko": 0.40,
        "directness": 0.35,
    },
    "Availability": {
        "examples_en": [
            "We should definitely do this again",
            "Are you free this weekend",
            "I'd love to continue this conversation",
            "Let me know when you're free",
        ],
        "examples_ko": [
            "언제 시간 돼", "주말에 뭐 해", "다음에 또 봐",
            "연락해", "카톡해", "문자해", "우리 또 만나자",
        ],
        "examples_es": [
            "¿Cuándo estás libre?",
            "Hablemos otra vez",
            "Avísame cuando puedas",
            "Podemos vernos",
        ],
        "examples_fr": [
            "Quand tu es libre ?",
            "On se reparle",
            "Dis-moi quand tu peux",
            "On peut se voir",
        ],
        "examples_de": [
            "Wann hast du Zeit?",
            "Melde dich, wenn du Zeit hast",
            "Wir können uns treffen",
            "Lass uns wieder reden",
        ],
        "examples_pt": [
            "Quando você está livre?",
            "Me avisa quando puder",
            "A gente pode se ver",
            "Vamos falar de novo",
        ],
        "intensity": 0.50,
        "intensity_ko": 0.50,
        "directness": 0.55,
    },
}

MUTUALITY_PATTERNS = [
    # English
    r"\b(?:we|us|our|together|both)\b",
    r"each other",
    r"let's",
    # Korean
    r"우리\s*(?:둘|같이|함께|는|가|도|끼리|만의|사이)",
    r"같이",
    r"함께",
    r"둘이서?",
    r"우리끼리",
    r"서로",
    # Spanish
    r"\b(?:nosotros|juntos|juntas|ambos)\b",
    # French
    r"\b(?:nous|ensemble|tous les deux)\b",
    # German
    r"\b(?:wir|zusammen|beide)\b",
    # Portuguese
    r"\b(?:nos|juntos|juntas|ambos)\b",
]

# Language adapters: high precision regex by language.
LANGUAGE_ADAPTERS: Dict[str, Dict[str, List[str]]] = {
    "ko": {
        "Affection": [r"좋아(?:해|합니다|해요|하는)", r"사랑(?:해|합니다|해요)", r"너라서\s*좋"],
        "Longing": [r"보고\s*싶", r"생각나", r"기다리"],
        "Care": [r"밥\s*먹", r"걱정", r"건강\s*챙", r"잘\s*자", r"잘\s*들어가", r"피곤하겠다", r"아프지\s*마"],
        "Future Together": [r"같이\s*(?:가자|보자|먹자|하자)", r"다음에\s*또", r"언제\s*시간"],
        "Adoration": [r"예쁘", r"멋있", r"귀엽", r"내\s*스타일"],
        "Intimacy": [r"자기야|자기\b|내\s*사랑|여보|우리\s*자기|아가"],
        "Physical Response": [r"설레|두근|심장\s*뛰|떨려|긴장돼|얼굴\s*빨개"],
        "Exclusivity": [r"너만|너밖에|특별해|처음이야|다른\s*사람.*아니"],
        "Curiosity": [r"더\s*알고\s*싶|궁금해|평소에\s*뭐\s*해|취미가\s*뭐야|어떤\s*사람"],
        "Playfulness": [r"ㅋㅋ+|ㅎㅎ+|장난이야|놀리는\s*거야|웃겨|귀엽게\s*구네"],
        "Availability": [r"연락해|카톡해|문자해|언제\s*봐|주말에\s*뭐\s*해|시간\s*돼"],
    },
    "en": {
        "Affection": [r"\bi\s+love\s+you\b", r"\bi\s+really\s+like\s+you\b", r"\bfallen\s+for\s+you\b"],
        "Longing": [r"\bmiss\s+you\b", r"\bcan'?t\s+wait\s+to\s+see\s+you\b", r"\bthinking\s+about\s+you\b"],
        "Care": [r"\btake\s+care\b", r"\bget\s+home\s+safe\b", r"\bworried\s+about\s+you\b"],
        "Future Together": [r"\blet'?s\b", r"\btogether\b", r"\bmeet\s+again\b", r"\bare\s+you\s+free\b"],
        "Adoration": [r"\bbeautiful\b", r"\bamazing\b", r"\bgorgeous\b", r"\battractive\b"],
        "Intimacy": [r"\b(?:sweetheart|honey|babe|darling|my love)\b"],
        "Physical Response": [r"\bbutterflies\b|\bheart\s+(?:racing|pounding)\b|\bnervous\s+around\s+you\b"],
        "Exclusivity": [r"\bonly\s+one\b|\bno\s+one\s+else\b|\bspecial\s+to\s+me\b"],
        "Curiosity": [r"\btell\s+me\s+more\b|\bwhat\s+do\s+you\s+do\s+for\s+fun\b|\bi'?d\s+love\s+to\s+know\s+more\b"],
        "Playfulness": [r"\byou'?re\s+so\s+funny\b|\bmaking\s+me\s+laugh\b|\bsuch\s+a\s+tease\b"],
        "Availability": [r"\bwhen\s+are\s+you\s+free\b|\blet\s+me\s+know\b|\bcontinue\s+this\s+conversation\b"],
    },
    "es": {
        "Affection": [r"te\s+quiero", r"te\s+amo", r"me\s+gustas"],
        "Longing": [r"te\s+extran", r"te\s+echo\s+de\s+menos"],
        "Care": [r"cu[ií]date", r"me\s+preocupa", r"descansa"],
        "Future Together": [r"vamos\s+juntos", r"nos\s+vemos\s+otra\s+vez", r"cuando\s+puedes"],
        "Adoration": [r"eres\s+hermos", r"eres\s+precios", r"incre[ií]ble"],
    },
    "fr": {
        "Affection": [r"je\s+t'aime", r"je\s+t'adore"],
        "Longing": [r"tu\s+me\s+manques"],
        "Care": [r"prends\s+soin", r"repose-toi"],
        "Future Together": [r"on\s+se\s+voit", r"ensemble", r"quand\s+tu\s+es\s+libre"],
        "Adoration": [r"tu\s+es\s+magnifique", r"tu\s+es\s+belle"],
    },
    "de": {
        "Affection": [r"ich\s+liebe\s+dich", r"ich\s+mag\s+dich"],
        "Longing": [r"ich\s+vermisse\s+dich"],
        "Care": [r"pass\s+auf\s+dich\s+auf", r"ruh\s+dich\s+aus"],
        "Future Together": [r"lass\s+uns\s+zusammen", r"sehen\s+wir\s+uns\s+wieder"],
        "Adoration": [r"du\s+bist\s+wundersch", r"du\s+bist\s+toll"],
    },
    "pt": {
        "Affection": [r"eu\s+te\s+amo", r"gosto\s+de\s+você", r"gosto\s+de\s+voce"],
        "Longing": [r"sinto\s+sua\s+falta"],
        "Care": [r"se\s+cuida", r"descansa"],
        "Future Together": [r"vamos\s+juntos", r"a\s+gente\s+se\s+vê\s+de\s+novo"],
        "Adoration": [r"você\s+é\s+linda", r"você\s+é\s+incrível", r"voce\s+e\s+linda"],
    },
}

# =============================================================================
# 언어별 Keyword Fallback 패턴
# =============================================================================

# 영어: 단어 단위 매칭 (원본 구조 유지)
EN_KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["love", "like you", "adore", "fallen for", "feelings for"],
    "Longing": ["miss", "can't wait", "wish you were", "thinking about you"],
    "Care": ["have you eaten", "take care", "worried about you", "get some rest"],
    "Future Together": ["together", "let's", "again", "next time", "are you free"],
    "Adoration": ["beautiful", "amazing", "gorgeous", "incredible", "attractive"],
    "Physical Response": ["heart racing", "butterflies", "nervous around", "flustered"],
    "Exclusivity": ["only one", "special to me", "never felt this", "unlike anyone"],
    "Intimacy": ["honey", "sweetheart", "darling", "babe", "my love"],
    "Curiosity": ["tell me more", "what do you do", "really interesting", "love to know"],
    "Playfulness": ["so funny", "making me laugh", "you're a tease", "so charming"],
    "Availability": ["are you free", "let me know when", "do this again"],
}

ES_KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["te quiero", "te amo", "me gustas", "me gustas mucho", "tengo sentimientos por ti"],
    "Longing": ["te extraño", "te extrano", "te echo de menos", "no puedo esperar", "pienso en ti"],
    "Care": ["has comido", "cuídate", "cuidate", "me preocupa", "descansa"],
    "Future Together": ["vamos juntos", "otra vez", "de nuevo", "cuando puedes", "este fin de semana"],
    "Adoration": ["hermosa", "guapa", "increíble", "preciosa", "atractiva"],
    "Physical Response": ["corazón", "mariposas", "nervioso", "me late"],
    "Exclusivity": ["solo tú", "sólo tú", "nadie como tú", "especial para mí"],
    "Intimacy": ["mi amor", "cariño", "corazón", "bebé", "mi vida"],
    "Curiosity": ["cuéntame más", "qué te gusta hacer", "quiero saber más", "eso es interesante"],
    "Playfulness": ["jaja", "me haces reír", "bromista", "divertido", "divertida"],
    "Availability": ["cuándo estás libre", "avísame", "podemos vernos", "otra vez"],
}

FR_KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["je t'aime", "je t'adore", "tu me plais", "j'ai des sentiments"],
    "Longing": ["tu me manques", "j'ai hâte de te voir", "je pense à toi"],
    "Care": ["tu as mangé", "prends soin de toi", "je m'inquiète", "repose-toi"],
    "Future Together": ["ensemble", "on se voit", "ce week-end", "refaire ça"],
    "Adoration": ["magnifique", "incroyable", "très belle", "superbe"],
    "Physical Response": ["mon cœur bat", "papillons", "nerveux", "je rougis"],
    "Exclusivity": ["tu es la seule", "personne comme toi", "spéciale pour moi"],
    "Intimacy": ["mon amour", "chéri", "chérie", "mon cœur", "bébé"],
    "Curiosity": ["dis-m'en plus", "tu fais quoi", "j'aimerais en savoir plus"],
    "Playfulness": ["tu me fais rire", "drôle", "taquin", "amusant"],
    "Availability": ["quand tu es libre", "on se reparle", "dis-moi quand tu peux"],
}

DE_KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["ich liebe dich", "ich mag dich", "gefühle für dich", "verliebt"],
    "Longing": ["ich vermisse dich", "ich denke an dich", "warte auf dich"],
    "Care": ["hast du gegessen", "pass auf dich auf", "ich mache mir sorgen", "ruh dich aus"],
    "Future Together": ["zusammen", "wiedersehen", "am wochenende frei", "nochmal"],
    "Adoration": ["wunderschön", "unglaublich", "du bist toll", "attraktiv"],
    "Physical Response": ["herz rast", "schmetterlinge", "nervös", "herz klopft"],
    "Exclusivity": ["nur du", "niemand wie du", "besonders für mich"],
    "Intimacy": ["schatz", "liebling", "mein herz", "babe"],
    "Curiosity": ["erzähl mir mehr", "was machst du gern", "interessant"],
    "Playfulness": ["du bringst mich zum lachen", "lustig", "witzig", "neckst mich"],
    "Availability": ["wann hast du zeit", "meld dich", "wir können uns treffen"],
}

PT_KEYWORD_MAP: Dict[str, List[str]] = {
    "Affection": ["eu te amo", "gosto de você", "sentimentos por você", "apaixonado"],
    "Longing": ["sinto sua falta", "saudade", "penso em você"],
    "Care": ["você já comeu", "se cuida", "estou preocupado", "descansa"],
    "Future Together": ["vamos juntos", "de novo", "fim de semana", "a gente se vê"],
    "Adoration": ["você é linda", "você é incrível", "maravilhosa", "atraente"],
    "Physical Response": ["coração acelera", "borboletas", "nervoso", "coração dispara"],
    "Exclusivity": ["só você", "ninguém como você", "especial para mim"],
    "Intimacy": ["meu amor", "meu bem", "querida", "querido", "bebê"],
    "Curiosity": ["me conta mais", "o que você gosta", "quero saber mais"],
    "Playfulness": ["você me faz rir", "engraçado", "divertido", "brincalhão"],
    "Availability": ["quando você está livre", "me avisa", "a gente pode se ver"],
}

LANG_KEYWORD_MAPS: Dict[str, Dict[str, List[str]]] = {
    "en": EN_KEYWORD_MAP,
    "es": ES_KEYWORD_MAP,
    "fr": FR_KEYWORD_MAP,
    "de": DE_KEYWORD_MAP,
    "pt": PT_KEYWORD_MAP,
}

# 한국어: 어간 기반 정규식 (어미 변화형 대응)
KO_REGEX_MAP: Dict[str, List[str]] = {
    "Affection": [
        r"좋아(?:해|했|하는|하거든|하는데|할|하고|합니다|해요|했어|한다|하게)",
        r"사랑(?:해|했|하는|해요|합니다)",
        r"빠진\s*것\s*같",
        r"좋아하게\s*된",
        r"너\s*(?:때문에|라서)\s*행복",
    ],
    "Longing": [
        r"보고\s*(?:싶어|파|싶다|싶은데|싶었어|싶었다)",
        r"자꾸\s*생각",
        r"(?:네|니)\s*생각",
        r"생각(?:나|났어|나더라|이\s*나)",
        r"기다리(?:고\s*있|고|는데|)",
        r"언제\s*(?:봐|와|만나|볼\s*수)",
    ],
    "Care": [
        r"밥\s*(?:먹었|먹었어|먹었어요|먹어|챙겨)",
        r"식사\s*(?:했|했어|했어요)",
        r"잘\s*(?:잤어|자|들어가|챙겨)",
        r"걱정(?:돼|됩니다|되는데)",
        r"(?:건강|몸)\s*챙겨",
        r"아프지\s*마",
        r"힘들(?:겠다|었겠다)",
        r"고생\s*했",
    ],
    "Adoration": [
        r"(?:정말|진짜|너무|완전)\s*(?:예쁘|멋있|귀엽|이쁘)",
        r"(?:예쁘|멋있|귀엽|이쁘)(?:다|네|는데|더라)",
        r"내\s*스타일",
        r"눈빛이\s*(?:예쁘|좋아)",
        r"웃을\s*때\s*(?:예쁘|멋있)",
        r"목소리가\s*(?:좋아|예뻐)",
    ],
    "Physical Response": [
        r"(?:심장이?|가슴이?)\s*(?:두근|뛰어|쿵쾅)",
        r"두근(?:두근|거려|거리는)",
        r"설레(?:는데|ㄴ다|는|어|네|었어)?",
        r"떨려",
        r"긴장(?:돼|됩니다)",
        r"얼굴이\s*빨개",
        r"나도\s*모르게\s*웃",
    ],
    "Intimacy": [
        r"자기(?:야|가|한테)?",
        r"내\s*사랑",
        r"여보(?:야)?",
        r"우리\s*자기",
    ],
    "Future Together": [
        r"같이\s*(?:가자|먹자|보자|하자|가고\s*싶|해보고\s*싶)",
        r"다음에\s*(?:또|는|봐|만나|가자)",
        r"또\s*(?:보자|만나자|가자)",
        r"언제\s*(?:시간\s*돼|볼\s*수\s*있어)",
        r"주말에\s*(?:뭐|어때|시간)",
        r"우리\s*(?:또|언제|다음에)",
    ],
    "Exclusivity": [
        r"너만\s*(?:이래|그래|봐|있으면)",
        r"너(?:밖에|한테만)",
        r"다른\s*사람한테는\s*(?:이러지|안)",
        r"처음\s*(?:이야|느끼는|느껴보는)",
        r"특별(?:해|한|하게)",
        r"이런\s*감정\s*처음",
    ],
    "Curiosity": [
        r"더\s*알고\s*싶어",
        r"어떻게\s*(?:됐어|된\s*거야)",
        r"뭐\s*좋아해",
        r"평소에\s*뭐",
        r"취미가\s*뭐야",
        r"진짜\s*궁금해",
        r"흥미롭다",
    ],
    "Playfulness": [
        r"ㅋㅋ",
        r"하하",
        r"웃겨",
        r"재밌다",
        r"장난이야",
        r"농담이야",
        r"귀엽게\s*구네",
    ],
    "Availability": [
        r"언제\s*시간\s*돼",
        r"주말에\s*뭐\s*해",
        r"다음에\s*(?:또|봐|만나자)",
        r"연락\s*해",
        r"카톡\s*해",
        r"우리\s*또",
    ],
}


_seed_cache: Dict[str, Dict[str, List[np.ndarray]]] = {}


def _normalize_conversation(conversation_obj) -> Dict:
    if isinstance(conversation_obj, dict) and "conversation" in conversation_obj:
        return conversation_obj
    conv = getattr(conversation_obj, "conversation", None)
    if conv is None:
        return {"conversation": []}
    normalized = []
    for u in conv:
        if isinstance(u, dict):
            normalized.append(u)
        else:
            normalized.append(
                {
                    "speaker": getattr(u, "speaker", None),
                    "start": getattr(u, "start", None),
                    "end": getattr(u, "end", None),
                    "text": getattr(u, "text", ""),
                }
            )
    return {"conversation": normalized}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _openai_embed(texts: List[str]) -> List[np.ndarray]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
        max_retries=1,
    )
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _check_mutuality(text: str) -> float:
    lowered = text.lower()
    return min(sum(1 for p in MUTUALITY_PATTERNS if re.search(p, lowered)) * 0.3, 1.0)


def _embedding_threshold(lang: str) -> float:
    return EMBEDDING_THRESHOLD_BY_LANG.get(lang, EMBEDDING_THRESHOLD_BY_LANG["default"])


def _decision_threshold(lang: str) -> float:
    return ROMANTIC_DECISION_THRESHOLD_BY_LANG.get(lang, ROMANTIC_DECISION_THRESHOLD_BY_LANG["default"])


def _get_intensity(cat: str, lang: str = "unknown") -> float:
    meta = ROMANTIC_SEED_EXAMPLES.get(cat, {})
    if lang == "ko" and "intensity_ko" in meta:
        return float(meta["intensity_ko"])
    return float(meta.get("intensity", 0.5))


def _get_directness(cat: str) -> float:
    return ROMANTIC_SEED_EXAMPLES.get(cat, {}).get("directness", 0.5)


def _get_seed_embeddings(lang: str) -> Dict[str, List[np.ndarray]]:
    lang_key = lang if lang in ("en", "ko", "es", "fr", "de", "pt") else "default"
    if lang_key in _seed_cache:
        return _seed_cache[lang_key]

    texts: List[str] = []
    owners: List[str] = []
    for cat, data in ROMANTIC_SEED_EXAMPLES.items():
        if lang_key == "ko":
            examples = data.get("examples_ko", [])
        elif lang_key == "en":
            examples = data.get("examples_en", [])
        elif lang_key == "es":
            examples = data.get("examples_es", [])
        elif lang_key == "fr":
            examples = data.get("examples_fr", [])
        elif lang_key == "de":
            examples = data.get("examples_de", [])
        elif lang_key == "pt":
            examples = data.get("examples_pt", [])
        else:
            examples = (
                data.get("examples_en", [])
                + data.get("examples_ko", [])
                + data.get("examples_es", [])
                + data.get("examples_fr", [])
                + data.get("examples_de", [])
                + data.get("examples_pt", [])
            )

        if not examples and lang_key in ("en", "ko", "es", "fr", "de", "pt"):
            examples = (
                data.get("examples_en", [])
                + data.get("examples_ko", [])
                + data.get("examples_es", [])
                + data.get("examples_fr", [])
                + data.get("examples_de", [])
                + data.get("examples_pt", [])
            )

        for ex in examples:
            texts.append(ex)
            owners.append(cat)

    if not texts:
        _seed_cache[lang_key] = {}
        return _seed_cache[lang_key]

    embeddings = _openai_embed(texts)
    cat_map: Dict[str, List[np.ndarray]] = {}
    for cat, emb in zip(owners, embeddings):
        cat_map.setdefault(cat, []).append(emb)
    _seed_cache[lang_key] = cat_map
    return cat_map


def _classify_utterance_embedding(emb: np.ndarray, seeds: Dict[str, List[np.ndarray]]) -> Tuple[str, float]:
    best_cat = "Neutral"
    best_score = 0.0
    for cat, emb_list in seeds.items():
        sims = [_cosine_similarity(emb, s) for s in emb_list]
        if not sims:
            continue
        score = 0.7 * max(sims) + 0.3 * float(np.mean(sims))
        if score > best_score:
            best_cat = cat
            best_score = score
    return best_cat, best_score


def _adapter_classify(text: str, lang: str) -> Tuple[str, float, str]:
    adapters = LANGUAGE_ADAPTERS.get(lang, {})
    for cat, patterns in adapters.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return cat, 0.78, "language_adapter"

    # Backoff to English adapter when latin script language is uncertain.
    if lang not in adapters and lang not in ("ko", "ja", "zh", "ar", "ru", "hi"):
        for cat, patterns in LANGUAGE_ADAPTERS.get("en", {}).items():
            for pat in patterns:
                if re.search(pat, text, flags=re.IGNORECASE):
                    return cat, 0.68, "language_adapter_backoff"

    return "Neutral", 0.0, "language_adapter"


def _keyword_match_en(lowered: str, keyword: str) -> bool:
    if " " in keyword or "'" in keyword:
        return keyword in lowered
    return re.search(r"\b" + re.escape(keyword) + r"\b", lowered) is not None


def _keyword_classify(text: str, lang: str) -> Tuple[str, float]:
    if lang == "ko":
        for cat, patterns in KO_REGEX_MAP.items():
            for pat in patterns:
                if re.search(pat, text):
                    return cat, 0.55
        return "Neutral", 0.0

    if lang not in LANG_KEYWORD_MAPS:
        return "Neutral", 0.0

    lowered = text.lower()
    for cat, keywords in LANG_KEYWORD_MAPS[lang].items():
        for kw in keywords:
            if lang == "en":
                if _keyword_match_en(lowered, kw):
                    return cat, 0.55
            else:
                if kw in lowered:
                    return cat, 0.55
    return "Neutral", 0.0


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    romantic = [r for r in rows if r["is_romantic"]]
    avg_intensity = float(np.mean([r["intensity"] for r in romantic])) if romantic else 0.0
    avg_directness = float(np.mean([r["directness"] for r in romantic])) if romantic else 0.0
    avg_mutuality = float(np.mean([r["mutuality"] for r in rows])) if rows else 0.0
    avg_conf = float(np.mean([r["confidence"] for r in rows])) if rows else 0.0
    coverage = len(romantic) / len(rows) if rows else 0.0

    cat_counts: Dict[str, int] = {}
    for r in romantic:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    overall = avg_intensity * (0.5 + 0.5 * min(coverage * 2, 1.0))
    return {
        "score": int(round(overall * 100)),
        "intensity": round(avg_intensity, 4),
        "directness": round(avg_directness, 4),
        "mutuality": round(avg_mutuality, 4),
        "coverage": round(coverage, 4),
        "confidence": round(avg_conf, 4),
        "category_distribution": cat_counts,
        "evidence_spans": [
            {
                "speaker": r["speaker"],
                "language": r["language"],
                "category": r["category"],
                "method": r["method"],
                "confidence": round(r["confidence"], 4),
                "evidence": r["evidence"],
            }
            for r in romantic
        ],
    }


def analyze_global(data: Dict[str, Any]) -> Dict[str, Any]:
    conversation = data.get("conversation", [])
    if not conversation:
        return {"score": 0, "method": "empty"}

    utterances = [u.get("text", "") for u in conversation if u.get("text")]
    speakers = [u.get("speaker") for u in conversation if u.get("text")]
    if len(set(filter(None, speakers))) < 2:
        return {"score": 0, "error": "need_2_speakers"}

    # utterance-level language for adapter routing.
    lang_meta = detect_utterance_languages([{"speaker": s, "text": t} for s, t in zip(speakers, utterances)])
    langs = [u.get("language", "unknown") for u in lang_meta.get("utterances", [])]

    emb_rows: List[Tuple[str, float]] = []
    emb_method = "embedding_unavailable"
    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            utt_embs = _openai_embed(utterances)
            for idx, emb in enumerate(utt_embs):
                lang = langs[idx] if idx < len(langs) else "unknown"
                seeds = _get_seed_embeddings(lang)
                emb_rows.append(_classify_utterance_embedding(emb, seeds))
            emb_method = "openai_embedding_bilingual"
        except Exception as exc:
            emb_rows = []
            emb_method = f"embedding_failed:{exc}"

    rows: List[Dict[str, Any]] = []
    for idx, (text, speaker) in enumerate(zip(utterances, speakers)):
        lang = langs[idx] if idx < len(langs) else "unknown"
        a_cat, a_conf, a_method = _adapter_classify(text, lang)
        k_cat, k_conf = _keyword_classify(text, lang)

        best_cat = "Neutral"
        best_conf = 0.0
        method = "keyword_fallback"

        if emb_rows:
            e_cat, e_conf = emb_rows[idx]
            e_valid = (e_cat in ROMANTIC_ALL_CATEGORIES) and (e_conf >= _embedding_threshold(lang))
            a_valid = (a_cat in ROMANTIC_ALL_CATEGORIES) and (a_conf > 0)

            if a_valid and e_valid and a_cat == e_cat:
                best_cat = a_cat
                best_conf = min(1.0, max(e_conf, a_conf) + 0.08)
                method = "adapter+embedding_agree"
            elif a_valid and (not e_valid or a_conf >= e_conf + 0.08):
                best_cat = a_cat
                best_conf = a_conf
                method = a_method
            elif e_valid:
                best_cat = e_cat
                best_conf = e_conf
                method = emb_method
            elif k_cat in ROMANTIC_ALL_CATEGORIES:
                best_cat = k_cat
                best_conf = k_conf
                method = "keyword_fallback"
        else:
            if a_cat in ROMANTIC_ALL_CATEGORIES:
                best_cat = a_cat
                best_conf = a_conf
                method = a_method
            elif k_cat in ROMANTIC_ALL_CATEGORIES:
                best_cat = k_cat
                best_conf = k_conf
                method = "keyword_fallback"

        category = best_cat if best_cat in ROMANTIC_ALL_CATEGORIES else "Neutral"
        is_romantic = category in ROMANTIC_CATEGORIES and best_conf >= _decision_threshold(lang)
        intensity = _get_intensity(category, lang=lang) * best_conf if is_romantic else 0.0
        directness = _get_directness(category) * best_conf if is_romantic else 0.0

        rows.append(
            {
                "speaker": speaker,
                "language": lang,
                "category": category,
                "confidence": best_conf,
                "intensity": round(intensity, 4),
                "directness": round(directness, 4),
                "mutuality": _check_mutuality(text),
                "is_romantic": is_romantic,
                "method": method,
                "evidence": text,
            }
        )

    result = _aggregate(rows)
    result["method"] = "hybrid_adapter_embedding"
    result["embedding_status"] = emb_method
    return result


class RomanticAnalyzer:
    def score(self, conversation_obj) -> Dict[str, Any]:
        data = _normalize_conversation(conversation_obj)
        conv = data.get("conversation", [])

        lang_stats = detect_conversation_language(conv)
        utterance_stats = detect_utterance_languages(conv)

        raw = analyze_global(data)
        raw["language_detected"] = {
            "conversation": lang_stats,
            "utterance_level": utterance_stats,
        }
        return {
            "scores": {"romantic_intent": float(raw.get("score", 0))},
            "raw": raw,
        }

"""
PokéGuru — AI Pokémon Mentor
FastAPI backend with full Sarvam AI suite:
  • Mayura      — Translation  (22 Indian languages)
  • Bulbul v3   — Text-to-Speech (Professor Oak speaks in your language)
  • Saaras v4   — Speech-to-Text (ask by voice)
  • sarvam-105b — LLM reasoning (Indian-optimised, primary)
  • Groq        — LLM fallback
  • PokéAPI     — Pokémon data
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import requests
import re
import json
import os
import random
import base64
from pathlib import Path
from groq import Groq

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_6fkkk48t_Ta1ykxEAac5bxNdmoZg9EW8P")
SARVAM_HEADERS = {
    "Content-Type": "application/json",
    "api-subscription-key": SARVAM_API_KEY,
}

# Sarvam API endpoints
SARVAM_TRANSLATE_URL  = "https://api.sarvam.ai/translate"
SARVAM_TTS_URL        = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_URL        = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHAT_URL       = "https://api.sarvam.ai/v1/chat/completions"

# Bulbul TTS speaker per language
TTS_SPEAKERS = {
    "te": "meera",    "hi": "anushka",  "ta": "pavithra",
    "kn": "suresh",   "ml": "nadia",    "mr": "kalyani",
    "bn": "arjun",    "gu": "manisha",  "en": "nadia",
}

# Groq fallback key (optional)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Language code map (BCP-47)
LANG_CODES = {
    "te": "te-IN",  # Telugu
    "hi": "hi-IN",  # Hindi
    "ta": "ta-IN",  # Tamil
    "kn": "kn-IN",  # Kannada
    "ml": "ml-IN",  # Malayalam
    "mr": "mr-IN",  # Marathi
    "bn": "bn-IN",  # Bengali
    "gu": "gu-IN",  # Gujarati
    "pa": "pa-IN",  # Punjabi
    "or": "or-IN",  # Odia
    "en": "en-IN",  # English
}

app = FastAPI(title="PokéGuru API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    message: str
    language: str = "te"       # default Telugu
    level: str = "beginner"    # "beginner" or "competitive"
    groq_key: str | None = None


class QueryResponse(BaseModel):
    original: str
    translated_input: str
    detected_pokemon: list[str]
    team_analysis: dict
    advice_english: str
    advice_native: str
    language: str


# ──────────────────────────────────────────────
# Sarvam AI Translation helpers
# ──────────────────────────────────────────────
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate using Sarvam Mayura model, chunking long texts to avoid character limits."""
    if not text or not text.strip():
        return text

    src_code = LANG_CODES.get(source_lang, "auto")
    tgt_code = LANG_CODES.get(target_lang, "en-IN")

    def _trans_single(chunk: str) -> str:
        body = {
            "input": chunk,
            "source_language_code": src_code,
            "target_language_code": tgt_code,
            "model": "mayura:v1",
            "mode": "modern-colloquial",
            "enable_preprocessing": True,
        }
        try:
            resp = requests.post(SARVAM_TRANSLATE_URL, headers=SARVAM_HEADERS, json=body, timeout=20)
            resp.raise_for_status()
            return resp.json().get("translated_text", chunk)
        except Exception as e:
            print(f"[Sarvam/Mayura error] {e}")
            return chunk

    if len(text) <= 850:
        return _trans_single(text)

    # Chunk text by paragraphs for natural multi-section translation
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    translated_paras = []
    for p in paragraphs:
        if len(p) <= 850:
            translated_paras.append(_trans_single(p))
        else:
            # Sub-split long paragraph by sentences if needed
            sentences = re.split(r'(?<=[.!?])\s+', p)
            sub_chunk = ""
            for s in sentences:
                if len(sub_chunk) + len(s) < 800:
                    sub_chunk += s + " "
                else:
                    if sub_chunk:
                        translated_paras.append(_trans_single(sub_chunk.strip()))
                    sub_chunk = s + " "
            if sub_chunk.strip():
                translated_paras.append(_trans_single(sub_chunk.strip()))

    return "\n\n".join(translated_paras) if translated_paras else text


def text_to_speech(text: str, lang: str) -> bytes | None:
    """
    Sarvam Bulbul v3 TTS — convert advice text to audio bytes (WAV).
    Returns raw audio bytes or None on failure.
    Bulbul accepts max ~500 chars per request; we chunk if longer.
    """
    tgt_code = LANG_CODES.get(lang, "en-IN")
    speaker  = TTS_SPEAKERS.get(lang, "nadia")

    # Chunk text into ≤500-char pieces on sentence boundaries
    chunks = []
    current = ""
    for sentence in re.split(r'(?<=[।.!?])\s+', text):
        if len(current) + len(sentence) < 490:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current.strip():
        chunks.append(current.strip())
    if not chunks:
        chunks = [text[:490]]

    audio_parts = []
    for chunk in chunks[:4]:  # max 4 chunks (~2000 chars)
        body = {
            "inputs": [chunk],
            "target_language_code": tgt_code,
            "speaker": speaker,
            "model": "bulbul:v3",
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
        }
        try:
            resp = requests.post(SARVAM_TTS_URL, headers=SARVAM_HEADERS, json=body, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Response: {"audios": ["<base64-wav>"]}
            audios = data.get("audios", [])
            if audios:
                audio_parts.append(base64.b64decode(audios[0]))
        except Exception as e:
            print(f"[Sarvam/Bulbul error] {e}")
    return b"".join(audio_parts) if audio_parts else None


def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm", lang: str = "auto") -> str | None:
    """
    Sarvam Saaras v4 STT — transcribe audio bytes to text.
    Returns transcript string or None on failure.
    """
    lang_code = LANG_CODES.get(lang, "unknown") if lang != "auto" else "unknown"
    files = {"file": (filename, audio_bytes, "audio/webm")}
    data  = {"model": "saaras:v4", "mode": "transcribe"}
    headers = {"api-subscription-key": SARVAM_API_KEY}  # no Content-Type (multipart)
    try:
        resp = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        return result.get("transcript")
    except Exception as e:
        print(f"[Sarvam/Saaras error] {e}")
        return None


def sarvam_llm_reason(system_prompt: str, user_prompt: str) -> str | None:
    """
    Sarvam 105B Conversations — State-of-the-art Indic-aware LLM reasoning.
    Uses the user's Sarvam API key directly.
    """
    body = {
        "model": "sarvam-105b-conversations",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 800,
        "temperature": 0.75,
    }
    try:
        resp = requests.post(SARVAM_CHAT_URL, headers=SARVAM_HEADERS, json=body, timeout=35)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content")
        if content and content.strip():
            return content.strip()
    except Exception as e:
        print(f"[Sarvam/105b chat error] {e}")
    return None




def detect_language(text: str) -> str:
    """Simple heuristic language detector by Unicode range."""
    # Telugu: U+0C00–U+0C7F
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "te"
    # Hindi/Devanagari: U+0900–U+097F
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    # Tamil: U+0B80–U+0BFF
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "ta"
    # Kannada: U+0C80–U+0CFF
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "kn"
    # Malayalam: U+0D00–U+0D7F
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "ml"
    # Bengali: U+0980–U+09FF
    if re.search(r'[\u0980-\u09FF]', text):
        return "bn"
    # Gujarati: U+0A80–U+0AFF
    if re.search(r'[\u0A80-\u0AFF]', text):
        return "gu"
    # Punjabi/Gurmukhi: U+0A00–U+0A7F
    if re.search(r'[\u0A00-\u0A7F]', text):
        return "pa"
    return "en"


# ──────────────────────────────────────────────
# Intent Detection — extract Pokémon names
# ──────────────────────────────────────────────
KNOWN_POKEMON = [
    "bulbasaur","ivysaur","venusaur","charmander","charmeleon","charizard",
    "squirtle","wartortle","blastoise","caterpie","metapod","butterfree",
    "weedle","kakuna","beedrill","pidgey","pidgeotto","pidgeot","rattata",
    "raticate","spearow","fearow","ekans","arbok","pikachu","raichu",
    "sandshrew","sandslash","nidoran","nidorina","nidoqueen","nidorino",
    "nidoking","clefairy","clefable","vulpix","ninetales","jigglypuff",
    "wigglytuff","zubat","golbat","oddish","gloom","vileplume","paras",
    "parasect","venonat","venomoth","diglett","dugtrio","meowth","persian",
    "psyduck","golduck","mankey","primeape","growlithe","arcanine","poliwag",
    "poliwhirl","poliwrath","abra","kadabra","alakazam","machop","machoke",
    "machamp","bellsprout","weepinbell","victreebel","tentacool","tentacruel",
    "geodude","graveler","golem","ponyta","rapidash","slowpoke","slowbro",
    "magnemite","magneton","farfetchd","doduo","dodrio","seel","dewgong",
    "grimer","muk","shellder","cloyster","gastly","haunter","gengar",
    "onix","drowzee","hypno","krabby","kingler","voltorb","electrode",
    "exeggcute","exeggutor","cubone","marowak","hitmonlee","hitmonchan",
    "lickitung","koffing","weezing","rhyhorn","rhydon","chansey","tangela",
    "kangaskhan","horsea","seadra","goldeen","seaking","staryu","starmie",
    "mrmime","scyther","jynx","electabuzz","magmar","pinsir","tauros",
    "magikarp","gyarados","lapras","ditto","eevee","vaporeon","jolteon",
    "flareon","porygon","omanyte","omastar","kabuto","kabutops","aerodactyl",
    "snorlax","articuno","zapdos","moltres","dratini","dragonair","dragonite",
    "mewtwo","mew",
    # Gen 2
    "chikorita","bayleef","meganium","cyndaquil","quilava","typhlosion",
    "totodile","croconaw","feraligatr","sentret","furret","hoothoot","noctowl",
    "ledyba","ledian","spinarak","ariados","crobat","chinchou","lanturn",
    "pichu","cleffa","igglybuff","togepi","togetic","natu","xatu","mareep",
    "flaaffy","ampharos","bellossom","marill","azumarill","sudowoodo",
    "politoed","hoppip","skiploom","jumpluff","aipom","sunkern","sunflora",
    "yanma","wooper","quagsire","espeon","umbreon","murkrow","slowking",
    "misdreavus","unown","wobbuffet","girafarig","pineco","forretress",
    "dunsparce","gligar","steelix","snubbull","granbull","qwilfish","scizor",
    "shuckle","heracross","sneasel","teddiursa","ursaring","slugma","magcargo",
    "swinub","piloswine","corsola","remoraid","octillery","delibird","mantine",
    "skarmory","houndour","houndoom","kingdra","phanpy","donphan","porygon2",
    "stantler","smeargle","tyrogue","hitmontop","smoochum","elekid","magby",
    "miltank","blissey","raikou","entei","suicune","larvitar","pupitar",
    "tyranitar","lugia","hooh","celebi",
    # Gen 3 starters + popular
    "treecko","grovyle","sceptile","torchic","combusken","blaziken",
    "mudkip","marshtomp","swampert","gardevoir","breloom","slaking",
    "ninjask","shedinja","hariyama","aggron","flygon","altaria","milotic",
    "metagross","latias","latios","kyogre","groudon","rayquaza","deoxys",
    # Gen 4 popular
    "turtwig","grotle","torterra","chimchar","monferno","infernape",
    "piplup","prinplup","empoleon","lucario","garchomp","togekiss","leafeon",
    "glaceon","gliscor","mamoswine","gallade","dialga","palkia","giratina",
    "cresselia","darkrai","arceus",
    # Gen 5 popular
    "snivy","servine","serperior","tepig","pignite","emboar",
    "oshawott","dewott","samurott","zoroark","reshiram","zekrom","kyurem",
    # Gen 6+
    "greninja","talonflame","sylveon","hawlucha","xerneas","yveltal",
    "zygarde","decidueye","incineroar","primarina","mimikyu","toxapex",
    "kommo-o","solgaleo","lunala","necrozma",
    # Gen 8
    "grookey","thwackey","rillaboom","scorbunny","raboot","cinderace",
    "sobble","drizzile","inteleon","zacian","zamazenta","eternatus",
    # Gen 9
    "sprigatito","floragato","meowscarada","fuecoco","crocalor","skeledirge",
    "quaxly","quaxwell","quaquaval","koraidon","miraidon",
]


def extract_pokemon_names(text: str) -> list[str]:
    """Extract Pokémon names from text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for name in KNOWN_POKEMON:
        # word boundary match
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            found.append(name)
    return list(dict.fromkeys(found))  # preserve order, deduplicate


# ──────────────────────────────────────────────
# PokéAPI helpers
# ──────────────────────────────────────────────
POKEAPI_BASE = "https://pokeapi.co/api/v2"

_pokemon_cache: dict = {}

def fetch_pokemon(name: str) -> dict | None:
    """Fetch Pokémon data from PokéAPI (with simple in-memory cache)."""
    name = name.lower()
    if name in _pokemon_cache:
        return _pokemon_cache[name]
    try:
        resp = requests.get(f"{POKEAPI_BASE}/pokemon/{name}", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        compact = {
            "name": data["name"],
            "id": data["id"],
            "types": [t["type"]["name"] for t in data["types"]],
            "stats": {s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
            "abilities": [a["ability"]["name"] for a in data["abilities"]],
            "sprite": data["sprites"]["front_default"],
            "official_art": (
                data["sprites"]
                    .get("other", {})
                    .get("official-artwork", {})
                    .get("front_default", data["sprites"]["front_default"])
            ),
        }
        _pokemon_cache[name] = compact
        return compact
    except Exception as e:
        print(f"[PokéAPI error] {name}: {e}")
        return None


# Type effectiveness chart (attacker type → {defender type: multiplier})
TYPE_CHART = {
    "normal":   {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire":     {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water":    {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass":    {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice":      {"water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "rock": 2, "dark": 2, "steel": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "ghost": 0, "fairy": 0.5},
    "poison":   {"grass": 2, "fairy": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0},
    "ground":   {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying":   {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic":  {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug":      {"fire": 0.5, "grass": 2, "fighting": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock":     {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost":    {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon":   {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark":     {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel":    {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "fairy": 2, "steel": 0.5, "grass": 0.5, "psychic": 0.5, "dragon": 0.5, "normal": 0.5, "flying": 0.5, "bug": 0.5, "poison": 0},
    "fairy":    {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}

ALL_TYPES = list(TYPE_CHART.keys())


def get_weaknesses(types: list[str]) -> dict[str, float]:
    """Return a dict of {attacker_type: multiplier} for a given defender's types."""
    multipliers: dict[str, float] = {}
    for attacker in ALL_TYPES:
        mult = 1.0
        for defender_type in types:
            chart = TYPE_CHART.get(attacker, {})
            mult *= chart.get(defender_type, 1.0)
        if mult != 1.0:
            multipliers[attacker] = mult
    return multipliers


def analyze_team(team_data: list[dict]) -> dict:
    """Compute team coverage, weaknesses, and missing types."""
    covered_types = set()
    all_weaknesses: dict[str, float] = {}
    pokemon_types_map = {}

    for mon in team_data:
        types = mon["types"]
        pokemon_types_map[mon["name"]] = types
        covered_types.update(types)

        weaknesses = get_weaknesses(types)
        for wtype, mult in weaknesses.items():
            if mult > 1.0:
                all_weaknesses[wtype] = max(all_weaknesses.get(wtype, 0), mult)

    # Severe weaknesses (4x = full team has it)
    severe = {t: m for t, m in all_weaknesses.items() if m >= 4}
    moderate = {t: m for t, m in all_weaknesses.items() if 1 < m < 4}

    # Missing offensive types (not in team)
    missing_coverage = [t for t in ALL_TYPES if t not in covered_types]

    # Recommended types to fill gaps
    recommended_types = []
    for wtype in severe:
        # Find types that resist the severe weakness
        for attacker, chart in TYPE_CHART.items():
            if chart.get(wtype, 1.0) >= 2 and attacker not in covered_types:
                if attacker not in recommended_types:
                    recommended_types.append(attacker)

    return {
        "team_types": list(covered_types),
        "severe_weaknesses": severe,
        "moderate_weaknesses": moderate,
        "missing_coverage": missing_coverage,
        "recommended_types": recommended_types,
        "pokemon_types": pokemon_types_map,
    }


# ──────────────────────────────────────────────
# LLM reasoning — Professor Oak persona
# ──────────────────────────────────────────────
def get_professor_oak_advice(
    team_info: list[dict],
    analysis: dict,
    user_query_english: str,
    level: str = "beginner",
    groq_key: str | None = None,
) -> str:
    """Generate strategic, varied advice using Sarvam 105B LLM, Groq, or dynamic fallback."""

    team_summary = []
    for mon in team_info:
        team_summary.append(
            f"- {mon['name'].capitalize()}: Types={mon['types']}, "
            f"HP={mon['stats'].get('hp')}, Atk={mon['stats'].get('attack')}, "
            f"SpAtk={mon['stats'].get('special-attack')}, "
            f"Def={mon['stats'].get('defense')}, Spd={mon['stats'].get('speed')}, "
            f"Abilities={mon['abilities']}"
        )

    system_prompt = """You are Professor Oak, the world's most renowned Pokémon researcher.
You speak with infectious passion, scholarly wisdom, and tactical genius.
Analyze the trainer's team with deep, nuanced reasoning:
1. Team Archetype & Balance: Assess offensive/defensive balance, speed tiers, physical vs. special distribution, and hazard control.
2. Compound Vulnerabilities: Highlight dangerous common offensive threats or type hazards that threaten this specific team.
3. Tailored Roster Additions: Recommend 2-3 distinct, specific Pokémon by name. For each, describe its role (Physical Wall, Bulky Pivot, Setup Sweeper, etc.), ideal ability, and specifically why it patches THIS team's gaps.
4. Level Adjustment: If Beginner, focus on fun execution and accessible synergy. If Competitive, discuss EVs, entry hazards, pivot switches, and win conditions.
Format with clean bullet points and Professor Oak's signature encouraging tone. Keep response under 320 words."""

    team_text = "\n".join(team_summary)
    analysis_text = json.dumps({
        "current_types_covered": analysis["team_types"],
        "severe_weaknesses": list(analysis["severe_weaknesses"].keys()),
        "moderate_weaknesses": list(analysis["moderate_weaknesses"].keys()),
        "recommended_types_to_add": analysis["recommended_types"][:5],
    }, indent=2)

    user_prompt = f"""Trainer Query: "{user_query_english}"

Current Team Roster:
{team_text}

Computed Weakness & Type Analysis:
{analysis_text}

Play Level: {level.capitalize()}

Please give your complete Professor Oak strategic evaluation. Recommend 2 to 3 specific Pokémon by name with tactical justification."""

    # 1. Primary: Sarvam 105B LLM reasoning (uses active Sarvam API key)
    sarvam_res = sarvam_llm_reason(system_prompt, user_prompt)
    if sarvam_res:
        return sarvam_res

    # 2. Secondary: Groq LLM (if key provided in request or env)
    active_groq_key = groq_key or GROQ_API_KEY
    if active_groq_key:
        try:
            client = Groq(api_key=active_groq_key)
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=650,
                temperature=0.75,
            )
            content = chat.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            print(f"[Groq error] {e}")

    # 3. Dynamic multi-pool fallback (guarantees diverse, role-tailored Pokémon)
    return generate_fallback_advice(team_info, analysis, level)


def generate_fallback_advice(team_info: list[dict], analysis: dict, level: str) -> str:
    """
    Intelligent, dynamic rule-based Professor Oak advisor.
    Evaluates physical vs special balance, defensive roles, and samples
    from an expansive pool so it never gives the exact same Pokémon.
    """
    current_names = {m["name"].lower() for m in team_info}
    team_display = [m["name"].capitalize() for m in team_info]
    severe = list(analysis["severe_weaknesses"].keys())
    moderate = list(analysis["moderate_weaknesses"].keys())
    rec_types = analysis["recommended_types"]

    # Calculate physical vs special stat tendencies
    total_atk = sum(m["stats"].get("attack", 70) for m in team_info)
    total_spatk = sum(m["stats"].get("special-attack", 70) for m in team_info)
    total_def = sum(m["stats"].get("defense", 70) for m in team_info)
    total_spdef = sum(m["stats"].get("special-defense", 70) for m in team_info)

    needs_physical_attacker = total_spatk > (total_atk * 1.25)
    needs_physical_wall = total_def < total_spdef or total_def < (len(team_info) * 75)

    # Diverse pool of Pokémon per type categorized by role
    POKEMON_POOLS = {
        "water": [
            ("Swampert", "Water/Ground physical tank that grants full Electric immunity and handles Rock/Fire"),
            ("Gyarados", "Water/Flying physical sweeper with Intimidate and Dragon Dance setup"),
            ("Milotic", "Marvel Scale special wall with reliable recovery and status absorption"),
            ("Rotom-Wash", "Water/Electric levitating pivot with Volt Switch momentum and Ground immunity"),
            ("Gastrodon", "Storm Drain ability redirects Water attacks while providing sturdy Ground bulk"),
            ("Tentacruel", "Fast Water/Poison hazard spinner that absorbs Toxic Spikes and walls fairies"),
        ],
        "ground": [
            ("Excadrill", "Steel/Ground powerhouse with Rapid Spin hazard removal and massive physical Attack"),
            ("Hippowdon", "Massive physical wall with Sand Stream and Slack Off recovery"),
            ("Gliscor", "Poison Heal physical pivot with Ground/Flying typing immune to Electric and Ground"),
            ("Mamoswine", "Ice/Ground breaker with Ice Shard priority to counter Dragons and Flying types"),
            ("Donphan", "Sturdy physical wall with Stealth Rock and Rapid Spin utility"),
        ],
        "steel": [
            ("Corviknight", "Flying/Steel defensive titan with Defog, Roost, and Ground/Poison immunities"),
            ("Ferrothorn", "Grass/Steel hazard wall providing Leech Seed and resistances to 10 types"),
            ("Scizor", "Bug/Steel technician with lethal priority Bullet Punch and U-turn momentum"),
            ("Aegislash", "Stance Change master that switches between impenetrable shield and devastating blade"),
            ("Magnezone", "Magnet Pull trapper that eliminates opposing Steel walls"),
        ],
        "grass": [
            ("Ferrothorn", "Iron Barbs hazard setter with incredible defensive typing"),
            ("Rillaboom", "Grassy Surge physical attacker that provides passive terrain healing and priority Grassy Glide"),
            ("Breloom", "Technician physical attacker with 100% accurate Spore sleep control"),
            ("Amoonguss", "Regenerator defensive pivot that absorbs attacks and disrupts with Spore"),
            ("Whimsicott", "Prankster support with Tailwind and Encore disruption"),
        ],
        "fairy": [
            ("Clefable", "Magic Guard utility wall immune to hazard damage, burn, and poison chip"),
            ("Azumarill", "Huge Power Water/Fairy physical wallbreaker with Belly Drum sweep potential"),
            ("Togekiss", "Serene Grace flinch master with Nasty Plot special sweeping capability"),
            ("Grimmsnarl", "Prankster screen setter (Reflect/Light Screen) to safeguard fragile teammates"),
            ("Sylveon", "Pixelate Hyper Voice special cannon with high Special Defense bulk"),
        ],
        "fighting": [
            ("Lucario", "Adaptable mixed attacker that threatens Normal, Rock, Dark, and Ice"),
            ("Conkeldurr", "Guts physical juggernaut with Drain Punch and Mach Punch priority"),
            ("Urshifu", "Rapid-strike or Single-strike breaker that bypasses protective barriers"),
            ("Hawlucha", "Unburden sweeping terror that sweeps once its consumable item activates"),
        ],
        "dragon": [
            ("Dragonite", "Multiscale dragon that safely sets up Dragon Dance and sweeps"),
            ("Garchomp", "Rough Skin physical striker with ferocious Earthquake and Swords Dance"),
            ("Dragapult", "Infiltration speed demon that outspeeds nearly the entire metagame"),
            ("Hydreigon", "Levitating Dark/Dragon special cannon that pierces bulky psychic and ghost types"),
        ],
        "normal": [
            ("Snorlax", "Immense HP wall with Thick Fat reducing Fire and Ice damage to trivial scratches"),
            ("Blissey", "Unrivaled special sponge with Natural Cure, Soft-Boiled, and Aromatherapy support"),
            ("Porygon2", "Eviolite defensive juggernaut that walls both physical and special threats"),
        ],
        "rock": [
            ("Tyranitar", "Sand Stream colossus with immense physical strength and boosted special bulk"),
            ("Aerodactyl", "Blistering fast Rock/Flying lead with Taunt and Stealth Rock control"),
        ],
        "psychic": [
            ("Slowbro", "Regenerator physical wall with Teleport pivot and Calm Mind setup"),
            ("Alakazam", "Magic Guard speedster with Focus Sash and Focus Blast coverage"),
        ],
        "ice": [
            ("Weavile", "Speedy Dark/Ice assassin with Triple Axel and Knock Off utility"),
            ("Alolan Ninetales", "Snow Warning lead that sets Aurora Veil to halve all incoming damage"),
        ],
        "dark": [
            ("Tyranitar", "Devastating Sand Stream tank that crushes opposing Psychic and Ghost threats"),
            ("Bisharp", "Defiant physical attacker that punishes opponent stat drops with Sucker Punch"),
            ("Mandibuzz", "Overcoat defensive bird that removes hazards and walls physical sweepers"),
        ],
        "ghost": [
            ("Mimikyu", "Disguise ability guarantees one free turn of Swords Dance setup"),
            ("Chandelure", "Flash Fire special nuclear weapon with 145 base Special Attack"),
        ]
    }

    # Gather candidate recommendations
    candidates = []
    considered_types = rec_types if rec_types else severe + moderate + ["water", "ground", "steel", "fairy"]

    for rtype in considered_types:
        pool = POKEMON_POOLS.get(rtype, [])
        for name, reason in pool:
            if name.lower() not in current_names and name not in [c[0] for c in candidates]:
                score = 1.0
                if needs_physical_attacker and ("physical" in reason.lower() or "sweeper" in reason.lower()):
                    score += 0.8
                if needs_physical_wall and ("wall" in reason.lower() or "tank" in reason.lower()):
                    score += 0.8
                candidates.append((name, reason, score, rtype))

    # Sort candidates by tactical fit and sample top picks
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_tier = candidates[:6]
    selected = random.sample(top_tier, min(3, len(top_tier))) if len(top_tier) >= 3 else candidates[:3]

    advice = [
        f"Ah, {', '.join(team_display)}! A spirited assembly of companions, trainer! 🌟",
        "",
        "### Tactical Evaluation",
    ]

    if needs_physical_attacker:
        advice.append("• **Damage Profile:** Your team currently leans heavily on Special Attacks. Adding a hard-hitting **Physical Attacker** will prevent specially bulky walls from stalling you out.")
    if needs_physical_wall:
        advice.append("• **Defensive Structure:** Your roster is rich in speed and offense, but lacks dedicated defensive bulk to absorb heavy physical impacts.")

    if severe:
        advice.append(f"• **Hazard Alert:** Your lineup carries a notable vulnerability to **{', '.join(t.upper() for t in severe)}** moves. We must shore up these gaps immediately!")

    advice.append("")
    advice.append("### Recommended Additions to Complete Your Roster:")
    for name, reason, _, rtype in selected:
        advice.append(f"• **{name}** ({rtype.capitalize()} synergy) — {reason}.")

    advice.append("")
    if level.lower() == "beginner":
        advice.append("🌱 **Beginner Guidance:** Focus on rotating your Pokémon into battle when their types resist incoming moves. Having one durable Pokémon you can safely switch into makes your journey ten times smoother!")
    else:
        advice.append("⚔️ **Competitive Guidance:** Pay close attention to entry hazards like Stealth Rock and speed tiers. Use U-turn or Volt Switch pivots to safely position your frail sweepers into favorable matchups.")

    advice.append("")
    advice.append("Remember: a well-rounded team triumphs through teamwork and synergy! 🔬")
    return "\n".join(advice)



# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "PokéGuru"}


@app.post("/api/query", response_model=QueryResponse)
def query_pokemon_guru(req: QueryRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 1: Detect language
    detected_lang = detect_language(message)
    lang = req.language if req.language != "auto" else detected_lang

    # Step 2: Translate input to English (for intent detection)
    if lang != "en":
        english_text = translate_text(message, source_lang=lang, target_lang="en")
    else:
        english_text = message

    print(f"[Pipeline] Original: {message[:80].encode('ascii', errors='replace').decode()}")
    print(f"[Pipeline] English:  {english_text[:80]}")

    # Step 3: Extract Pokémon names (from BOTH original + translated)
    detected_from_original = extract_pokemon_names(message)
    detected_from_english = extract_pokemon_names(english_text)
    detected = list(dict.fromkeys(detected_from_original + detected_from_english))

    print(f"[Pipeline] Detected: {detected}")

    if not detected:
        # Return a friendly message even if no Pokémon found
        no_poke_msg = "నేను మీ మెసేజ్‌లో ఏ Pokémon పేరూ గుర్తించలేకపోయాను. దయచేసి Pokémon పేర్లు చెప్పండి! (e.g., Charizard, Pikachu)"
        if lang != "te":
            no_poke_msg = "I couldn't detect any Pokémon names in your message. Please mention Pokémon names like Charizard, Pikachu, Gengar!"
        return QueryResponse(
            original=message,
            translated_input=english_text,
            detected_pokemon=[],
            team_analysis={},
            advice_english="Please mention Pokémon names in your message! For example: 'My team has Charizard, Pikachu, Gengar. What should I add?'",
            advice_native=no_poke_msg,
            language=lang,
        )

    # Step 4: Fetch Pokémon data from PokéAPI
    team_data = []
    for name in detected:
        data = fetch_pokemon(name)
        if data:
            team_data.append(data)

    if not team_data:
        raise HTTPException(status_code=404, detail="Could not fetch data for any detected Pokémon")

    # Step 5: Analyze team
    analysis = analyze_team(team_data)

    # Step 6: LLM reasoning (Professor Oak)
    advice_english = get_professor_oak_advice(
        team_data, analysis, english_text, level=req.level, groq_key=req.groq_key
    )

    # Step 7: Translate advice back to native language
    if lang != "en":
        advice_native = translate_text(advice_english, source_lang="en", target_lang=lang)
    else:
        advice_native = advice_english

    return QueryResponse(
        original=message,
        translated_input=english_text,
        detected_pokemon=detected,
        team_analysis={
            "types_covered": analysis["team_types"],
            "severe_weaknesses": list(analysis["severe_weaknesses"].keys()),
            "moderate_weaknesses": list(analysis["moderate_weaknesses"].keys()),
            "recommended_types": analysis["recommended_types"][:5],
            "pokemon_details": [
                {
                    "name": m["name"],
                    "types": m["types"],
                    "sprite": m["sprite"],
                    "official_art": m["official_art"],
                    "stats": m["stats"],
                    "abilities": m["abilities"],
                }
                for m in team_data
            ],
        },
        advice_english=advice_english,
        advice_native=advice_native,
        language=lang,
    )


@app.get("/api/pokemon/{name}")
def get_pokemon_info(name: str):
    """Get raw Pokémon data by name."""
    data = fetch_pokemon(name.lower())
    if not data:
        raise HTTPException(status_code=404, detail=f"Pokémon '{name}' not found")
    return data


# ──────────────────────────────────────────────
# Bulbul TTS endpoint
# ──────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    language: str = "te"


@app.post("/api/tts")
def tts_endpoint(req: TTSRequest):
    """
    Convert text to speech using Sarvam Bulbul v3.
    Returns audio as base64-encoded WAV inside JSON so the
    browser can decode and play it without CORS audio issues.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Trim to reasonable length (Bulbul handles ~500 chars per chunk internally)
    text = req.text.strip()[:2000]

    audio_bytes = text_to_speech(text, lang=req.language)
    if not audio_bytes:
        raise HTTPException(
            status_code=502,
            detail="Bulbul TTS failed — check Sarvam API key or try again"
        )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {
        "audio_b64": audio_b64,
        "format": "wav",
        "language": req.language,
        "speaker": TTS_SPEAKERS.get(req.language, "nadia"),
        "model": "bulbul:v3",
        "char_count": len(text),
    }


# ──────────────────────────────────────────────
# Saaras STT endpoint
# ──────────────────────────────────────────────
@app.post("/api/stt")
async def stt_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
):
    """
    Transcribe uploaded audio using Sarvam Saaras v4.
    Returns the transcript text.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    transcript = speech_to_text(audio_bytes, filename=file.filename or "audio.webm", lang=language)
    if transcript is None:
        raise HTTPException(status_code=502, detail="Saaras STT failed — try again")

    return {
        "transcript": transcript,
        "language": language,
        "model": "saaras:v4",
    }


# ──────────────────────────────────────────────
# Serve frontend
# ──────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "PokéGuru API running. Frontend not found."}


# Mount the entire frontend dir as static files (for any extra assets)
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

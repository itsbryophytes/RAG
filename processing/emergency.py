import re
from utils.logger import get_logger

logger = get_logger(__name__)

TIER1_PATTERNS = [
    r"\b(chest\s*pain|heart\s*attack|cardiac\s*arrest)\b",
    r"\b(can'?t\s*breathe|difficulty\s*breath|shortness\s*of\s*breath)\b",
    r"\b(stroke|face\s*drooping|arm\s*weak|speech\s*difficult)\b",
    r"\b(suicid|kill\s*(my)?self|end\s*my\s*life|want\s*to\s*die)\b",
    r"\b(overdos|took\s*too\s*many\s*(pill|tablet|medication))\b",
    r"\b(unconscious|not\s*breathing|stopped\s*breathing)\b",
    r"\b(severe\s*bleeding|blood\s*everywhere|losing\s*a\s*lot\s*of\s*blood)\b",
    r"\b(anaphylaxis|throat\s*closing|tongue\s*swelling)\b",
    r"\b(seizure|fitting|convuls)\b",
    r"\b(dada\s*(sakit|nyeri)|nyeri\s*dada)\b",
    r"\b(sesak\s*napas|sulit\s*bernapas|tidak\s*bisa\s*bernapas)\b",
    r"\b(pingsan|tidak\s*sadar|tidak\s*bangun)\b",
    r"\b(kejang|kejang-kejang)\b",
    r"\b(jantung\s*berhenti|henti\s*jantung)\b",
    r"\b(muntah\s*darah|batuk\s*darah)\b",
    r"\b(perdarahan\s*hebat|darah\s*tidak\s*berhenti)\b",
    r"\b(bibir\s*(biru|kebiruan))\b",
    r"\b(kesulitan\s*berbicara|bicara\s*(pelo|tidak\s*jelas))\b",
    r"\b(wajah\s*(mencong|drooping))\b",
    r"\b(lengan\s*(lemah|mati\s*rasa))\b",
]

TIER2_PATTERNS = [
    r"\b(high\s*fever|fever\s*over\s*4[01]|temperature\s*of\s*4[01])\b",
    r"\b(severe\s*(pain|headache|stomach\s*pain|abdominal\s*pain))\b",
    r"\b(coughing\s*blood|blood\s*in\s*urine|rectal\s*bleeding)\b",
    r"\b(sudden\s*(vision|hearing)\s*loss)\b",
    r"\b(confusion|disoriented|altered\s*consciousness)\b",
    r"\b(diabetic\s*crisis|hypoglycemi|blood\s*sugar\s*(very\s*)?(low|high))\b",
    r"\b(demam\s*tinggi|panas\s*tinggi)\b",
    r"\b(sakit\s*kepala\s*parah)\b",
    r"\b(sakit\s*perut\s*parah|nyeri\s*perut\s*parah)\b",
    r"\b(mual\s*parah|muntah\s*terus)\b",
    r"\b(lemas\s*berat)\b",
    r"\b(pusing\s*berat)\b",
    r"\b(pandangan\s*(kabur|gelap))\b",
    r"\b(darah\s*(di\s*urin|di\s*feses|bab\s*berdarah))\b",
    r"\b(tiba-tiba\s*(buta|tuli))\b",
    r"\b(gula\s*darah\s*(tinggi|rendah))\b",
]

TIER1_REGEX = re.compile("|".join(TIER1_PATTERNS), re.IGNORECASE)
TIER2_REGEX = re.compile("|".join(TIER2_PATTERNS), re.IGNORECASE)

EMERGENCY_RESPONSE_TIER1 = """\
🚨 **EMERGENCY DETECTED**

Based on what you've described, this may be a medical emergency.

**Please call your local emergency number (e.g. 112 / 119 / 911) immediately.**

Do NOT rely on this app during an emergency. I am an AI assistant and cannot provide real-time medical assistance.

If you are with someone who is unresponsive or in danger, call emergency services NOW.
"""

EMERGENCY_RESPONSE_TIER2 = """\
⚠️ **URGENT MEDICAL ATTENTION RECOMMENDED**

The symptoms you described may require prompt medical evaluation.

**Please contact your doctor or visit an urgent care / emergency room as soon as possible.**

Do not delay seeking care based on information from this app. I am an AI and cannot examine you or provide a diagnosis.
"""

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def check_emergency(text: str) -> tuple[bool, int, str | None]:
    text = normalize(text)
    
    if TIER1_REGEX.search(text):
        logger.warning("TIER-1 emergency detected in user message")
        return True, 1, EMERGENCY_RESPONSE_TIER1

    if TIER2_REGEX.search(text):
        logger.warning("TIER-2 urgency detected in user message")
        return True, 2, EMERGENCY_RESPONSE_TIER2

    return False, 0, None
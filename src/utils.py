
import re

EMOJI_RE = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # Reemplaza múltiples espacios por uno solo
    return s.lower()  # Convierte el texto a minúsculas


def looks_out_of_context(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return True
    s0 = s.strip()
    if len(s0) < 3:
        return True
    tmp = re.sub(r"@\w+", "", s0)
    tmp = re.sub(r"http\S+|www\.\S+", "", tmp)
    tmp = EMOJI_RE.sub("", tmp)
    tmp = re.sub(r"\s+", "", tmp)
    return len(tmp) == 0

def normalize_brand(token: str) -> str:
    t = token.lower().strip()
    if t in {"tigo", "@tigoparaguay", "#tigo"}:
        return "Tigo"
    if t in {"personal", "@personalpy", "#personal"}:
        return "Personal"
    if t in {"claro", "@claropy", "#claro"}:
        return "Claro"
    if t in {"vox", "copaco", "@copacopy", "@copaco_sa", "@voxpy", "#vox", "#copaco"}:
        return "Copaco"
    return ""

BRAND_PAT = re.compile(
    r"\b(tigo|personal|claro|copaco|vox)\b|"
    r"(@tigoparaguay|@personalpy|@claropy|@copacopy|@copaco_sa|@voxpy)|"
    r"(#tigo|#personal|#claro|#copaco|#vox)",
    re.I
)

def detect_brand(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = BRAND_PAT.findall(text.lower())
    tokens = []
    for tup in m:
        if isinstance(tup, tuple):
            for tok in tup:
                if tok:
                    tokens.append(tok)
        elif isinstance(tup, str):
            tokens.append(tup)
    for token in tokens:
        b = normalize_brand(token)
        if b:
            return b
    return ""

def safe_year(dt):
    try:
        y = int(dt.year)
        return y
    except Exception:
        return None

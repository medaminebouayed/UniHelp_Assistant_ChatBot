"""Test augmentation logic standalone."""
import re

_FR_QUERY_FIXES = {
    "accreditations": "accr\u00e9ditations",
    "accreditation": "accr\u00e9ditation",
    "accredite": "accr\u00e9dite",
    "accredites": "accr\u00e9dites",
    "mobilite": "mobilit\u00e9",
    "mobilites": "mobilit\u00e9s",
    "scolarite": "scolarit\u00e9",
    "universite": "universit\u00e9",
    "diplome": "dipl\u00f4me",
    "diplomes": "dipl\u00f4mes",
    "etudiant": "\u00e9tudiant",
    "etudiants": "\u00e9tudiants",
    "ecole": "\u00e9cole",
    "ecoles": "\u00e9coles",
    "faculte": "facult\u00e9",
    "facultes": "facult\u00e9s",
    "programme": "programme",
    "programmes": "programmes",
    "stage": "stage",
    "stages": "stages",
    "generale": "g\u00e9n\u00e9rale",
    "mecanique": "m\u00e9canique",
    "electronique": "\u00e9lectronique",
    "informatique": "informatique",
    "industrie": "industrie",
    "industriel": "industriel",
    "prepa": "pr\u00e9pa",
    "mastere": "mast\u00e8re",
}

def _augment_query(question):
    q = question.strip()
    if len(q) < 3:
        return q
    
    tokens = re.findall(r"[A-Za-z\xc0-\xff0-9]+", q)
    corrections = []
    for tok in tokens:
        tok_low = tok.lower()
        if tok_low in _FR_QUERY_FIXES:
            fix = _FR_QUERY_FIXES[tok_low]
            if tok != fix and tok_low != fix:
                corrections.append((tok, fix))
    
    if not corrections:
        return q
    
    result = q
    for old, new in corrections:
        result = re.sub(r'\b' + re.escape(old) + r'\b', new, result, flags=re.IGNORECASE)
    
    return result

queries = [
    "C'est quoi les frais de scolarite a IIT ?",
    "accreditations IIT",
    "frais scolarite",
    "programme mobilite ISB",
    "Comment se deroule le processus d'inscription a ISB ?",
]

for q in queries:
    result = _augment_query(q)
    changed = " <- CHANGED" if result != q else ""
    print(f"  '{q}' -> '{result}'{changed}")

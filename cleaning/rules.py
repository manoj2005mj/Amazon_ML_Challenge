"""Static tables for the cleaning stage.

Everything here is fixed knowledge (legal forms, abbreviations, state and
region names) or a pattern read off the challenge data itself. No external
lookups are performed anywhere in the cleaning stage.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def ascii_fold(s: str) -> str:
    """Transliterate to ASCII (static unidecode table) and squeeze spaces."""

    return _WS.sub(" ", unidecode(nfkc(s))).strip()


_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def key(s: str) -> str:
    """Lookup key: transliterated, case-folded, punctuation -> space."""

    if not s:
        return ""
    if s.isascii():
        return _WS.sub(" ", _NON_ALNUM.sub(" ", s.lower())).strip()
    return _WS.sub(" ", _NON_ALNUM.sub(" ", ascii_fold(s).casefold())).strip()


# --------------------------------------------------------------------------
# Placeholders (component level)
# --------------------------------------------------------------------------

PLACEHOLDERS = frozenset(
    {"null", "<null>", "none", "n/a", "na", "nan", "nil", "-", "--", ".", "unknown",
     "not available", "not applicable", "undefined", "n.a.", "n.a", "no address"}
)

# --------------------------------------------------------------------------
# Legal forms
# --------------------------------------------------------------------------

#: canonical -> spellings (already lower-case, dots removed, single tokens or
#: two-token phrases). ``PVT LTD`` is kept as two separate canonicals so the
#: sidecar reads ``LTD+PVT`` for "Private Limited".
LEGAL_FORMS: dict[str, tuple[str, ...]] = {
    "INC": ("inc", "incorporated", "incorporation"),
    "CORP": ("corp", "corporation", "corpn"),
    "CO": ("co", "company", "cie", "compagnie"),
    "LLC": ("llc", "lc", "l l c"),
    "LLP": ("llp", "l l p", "elelpi", "elelpii", "ailailpii"),
    "LP": ("lp",),
    "PLC": ("plc",),
    "LTD": ("ltd", "limited", "limted", "limitd", "limitted", "limittedd", "limttidd", "limittett", "limitedd", "limitedh"),
    "PVT": ("pvt", "private", "prvt", "pvt ltd", "praivet", "priavate", "privte", "praaivett", "praaiiveett", "praaibhett", "praiveett", "praivett"),
    "PTE": ("pte",),
    "PC": ("pc", "pllc", "p c"),
    "GMBH": ("gmbh",),
    "SA": ("sa", "s a"),
    "SARL": ("sarl", "s a r l"),
    "SAS": ("sas", "s a s"),
    "SASU": ("sasu", "s a s u"),
    "EURL": ("eurl", "e u r l"),
    "SCI": ("sci", "s c i"),
    "SNC": ("snc", "s n c"),
    "EI": ("ei",),
    "SCOP": ("scop",),
}
LEGAL_CANON: dict[str, str] = {v: k for k, vs in LEGAL_FORMS.items() for v in vs}

#: Forms that may be removed from anywhere in the name (unambiguous).
LEGAL_ANYWHERE = frozenset(
    {"INC", "CORP", "LLC", "LLP", "PLC", "LTD", "PVT", "PTE", "GMBH",
     "SARL", "SAS", "SASU", "EURL", "SCI", "SNC", "SCOP"}
)
#: Forms only removed at the end of the name (they are also ordinary words or
#: initials: "LP Gas Agency", "PC Jeweller", "Co-op", "SA Tennis").
LEGAL_END_ONLY = frozenset({"CO", "SA", "EI", "LP", "PC"})
#: In France the generator also moves the legal form to the front ("SA Comite",
#: "EI Dupont"); these end-only forms may be removed at index 0 there.
LEGAL_START_FR = frozenset({"SA", "EI"})

#: Devanagari / other-script legal forms, matched on the raw string before
#: transliteration (the dictionary learned from data catches the rest).
INDIC_LEGAL = {
    "प्राइवेट लिमिटेड": "PVT LTD", "प्रा. लि.": "PVT LTD", "प्रा.लि.": "PVT LTD", "प्रा लि": "PVT LTD",
    "प्राइवेट": "PVT", "लिमिटेड": "LTD", "एलएलपी": "LLP", "एलएलसी": "LLC", "इंक": "INC", "कॉर्प": "CORP",
    "ప్రైవేట్ లిమిటెడ్": "PVT LTD", "లిమిటెడ్": "LTD", "ప్రైవేట్": "PVT",
    "പ്രൈവറ്റ് ലിമിറ്റഡ്": "PVT LTD", "ലിമിറ്റഡ്": "LTD",
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್": "PVT LTD", "ಲಿಮಿಟೆಡ್": "LTD",
    "பிரைவேட் லிமிடெட்": "PVT LTD", "லிமிடெட்": "LTD",
    "પ્રાઇવેટ લિમિટેડ": "PVT LTD", "લિમિટેડ": "LTD",
    "প্রাইভেট লিমিটেড": "PVT LTD", "লিমিটেড": "LTD",
    "ପ୍ରାଇଭେଟ ଲିମିଟେଡ": "PVT LTD", "ਪ੍ਰਾਈਵੇਟ ਲਿਮਿਟੇਡ": "PVT LTD",
    # abbreviated / other-script spellings (review 2026-09-26)
    "प्रा. लि": "PVT LTD", "प्रा.लि": "PVT LTD", "प्रा लि.": "PVT LTD",
    "પ્રા. લિ.": "PVT LTD", "પ્રા.લિ.": "PVT LTD", "પ્રા. લિ": "PVT LTD", "લિ.": "LTD",
    "ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ": "PVT LTD", "ਲਿਮਟਿਡ": "LTD", "ਲਿਮਿਟੇਡ": "LTD", "ਲਿ.": "LTD", "ਐਲਐਲਪੀ": "LLP",
    "ପ୍ରାଇଭେଟ ଲିମିଟେଡ୍": "PVT LTD", "ଲିମିଟେଡ୍": "LTD", "ଲିମିଟେଡ": "LTD", "ପ୍ରାଇଭେଟ": "PVT",
    "எல்எல்பி": "LLP", "ఎల్ఎల్పీ": "LLP", "ಎಲ್ಎಲ್ಪಿ": "LLP", "എൽഎൽപി": "LLP", "এলএলপি": "LLP",
    "ਪ੍ਰਾਈਵੇਟ": "PVT", "প্রাইভেট": "PVT", "પ્રાઇવેટ": "PVT", "ప్రైవేటు": "PVT", "பிரைவேட்": "PVT",
}
#: Sort longest first so "प्राइवेट लिमिटेड" wins over "लिमिटेड".
INDIC_LEGAL_RX = re.compile("|".join(re.escape(k) for k in sorted(INDIC_LEGAL, key=len, reverse=True)))

# --------------------------------------------------------------------------
# Name noise
# --------------------------------------------------------------------------

#: Injected at a fixed ~1.3% rate each in Indian Source-2 names (Shree, Om, Jai, Sai
#: are genuine name words at other rates and are NOT stripped).
HONORIFICS = frozenset({"mr", "mrs", "dr", "smt", "shri", "sri", "m/s", "messrs", "mr.", "dr.", "mrs.", "smt.", "m/s."})
#: "a" / "an" are initials as often as articles ("A K Traders"); only "the" is safe.
ARTICLES = frozenset({"the"})
FR_ETS = {"ets": "Etablissements", "etab": "Etablissements", "établissements": "Etablissements", "etablissements": "Etablissements", "sté": "Societe", "ste": "Societe", "societe": "Societe", "société": "Societe"}

RX_ID_TAG = re.compile(r"\s*\(\s*id\s*:?\s*(\d+)\s*\)\s*", re.I)
RX_THE_TAG = re.compile(r"\s*\(\s*the\s*\)\s*$", re.I)
RX_COUNTRY_TAG = re.compile(r"\s*[\(\[]\s*(france|india|usa|us|u\.s\.a\.?|u\.s\.|uk|united states)\s*[\)\]]\s*", re.I)
RX_BRACKET_LEGAL = re.compile(r"\s*[\(\[]\s*([A-Za-z.]{2,12})\s*[\)\]]\s*")
#: "X formerly Y", "X dba Y", "X a/k/a Y", ... where X is the generator's pseudo-word
#: (>= 5 letters, optionally + Co/One/Labs/Sys/Group) and Y is the real name.
RX_ALIAS = re.compile(
    r"^\s*(?P<new>[A-Za-z][A-Za-z+\-]{4,}(?:\s+(?:co|one|labs|sys|group))?)\s+"
    r"(?:dba:?|d/b/a:?|d\.b\.a\.?:?|doing\s+business\s+as:?|a/?k/?a:?|a\.k\.a\.?:?|f/?k/?a:?|f\.k\.a\.?:?|n[ée]e:?|"
    r"formerly(?:\s+known\s+as)?:?|trading\s+as:?|t/a:?)\s+(?P<name>.+?)\s*$",
    re.I,
)
RX_WEB_SUFFIX = re.compile(r"\s*\|\s*(?:https?://)?(?:www\.)?\S+\.(?:com|net|org|in|co\.in|fr|io|biz|info)\S*\s*$", re.I)
RX_ZW = re.compile(r"[​‌‍﻿]")
RX_JUNK_EDGE = re.compile(r"^[\s\*\-\>\<\.\~\!\|\+\=\#\@\^\_\:\;\,\"\']+|[\s\*\-\>\<\.\~\!\|\+\=\#\@\^\_\:\;\,\"\']+$")
RX_WEB = re.compile(r"^(?:https?://)?(?:www\.)?(?P<host>[a-z0-9][a-z0-9\-\.]*\.(?:com|net|org|in|fr|co|io|biz|info|us|eu|co\.in|co\.uk))(?:/\S*)?$", re.I)
RX_WEB_ANY = re.compile(r"(https?://|www\.|\.(com|net|org|in|fr|co|io|biz|info)\b)", re.I)
RX_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?")  # S.A.S. / L.L.C. / P.C.
RX_POSSESSIVE = re.compile(r"(\w)['’`]s\b", re.I)
RX_APOS = re.compile(r"['’`]")
RX_NAME_PUNCT = re.compile(r"[\(\)\[\]\{\}/\\,;:\"“”«»|<>*_=~^]+")
RX_MULTI_HYPHEN = re.compile(r"\s*-\s*")

# --------------------------------------------------------------------------
# Addresses: numbers
# --------------------------------------------------------------------------

#: Number markers removed in front of a number: "H.No 44", "Door No 183", "D.No 25",
#: "Plot No. 17", "Flat 203", "No. 5", "N° 42", "Nos. 5-6", "House Number 5", "#42".
RX_NUM_MARKER = re.compile(
    r"\b(?:(?:[a-z]\.\s*){1,3}nos?\.?|[sdhfpwt]\.?nos?\.?|"
    r"(?:house|door|plot|flat|shop|office|gala|unit|site|survey|khasra|room|gate|ward|bldg|building|kh|sy|old|new|h)"
    r"\s*\.?\s*(?:nos?\.?|number|num)?|nos?\.?|num(?:ber|ero)?|n°|nº)\s*[:\-#]?\s*(?=[#\(]?\s*(?:\d|[A-Za-z]\s?-?\d))",
    re.I,
)
RX_DEGREE_NO = re.compile(r"\b[Nn]\s*[°º]\s*")
RX_HASH_NUM = re.compile(r"#+\s*(?=\d|[A-Za-z]-?\d)")
RX_PAREN_NUM = re.compile(r"\(\s*(\d+[A-Za-z]?)\s*\)")
RX_NUM_TRAIL = re.compile(r"\b(\d+)\s*[\-\.]\s+")  # "6445- Moors" / "906. Darnell" / "52 - Rue"
RX_LEAD_ZERO = re.compile(r"\b0+(\d{1,4})\b")  # 0029 -> 29 (never 5+ digit codes)
RX_BIS = re.compile(r"\b(\d+)\s*(bis|ter|quater|b|t)\b", re.I)
RX_HALF = re.compile(r"\b(\d+)\s+1/2\b")

# --------------------------------------------------------------------------
# Addresses: US
# --------------------------------------------------------------------------

US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "washington dc": "DC", "puerto rico": "PR",
}
US_STATE_ABBR = frozenset(US_STATES.values())

US_STREET: dict[str, str] = {
    "rd": "Road", "ave": "Avenue", "av": "Avenue", "blvd": "Boulevard", "blv": "Boulevard", "dr": "Drive", "drv": "Drive",
    "ln": "Lane", "ct": "Court", "crt": "Court", "cir": "Circle", "pl": "Place", "sq": "Square", "hwy": "Highway",
    "pkwy": "Parkway", "pky": "Parkway", "ter": "Terrace", "terr": "Terrace", "trl": "Trail", "tr": "Trail",
    "pt": "Point", "cv": "Cove", "xing": "Crossing", "ctr": "Center", "mt": "Mount", "ft": "Fort", "hts": "Heights",
    "jct": "Junction", "expy": "Expressway", "fwy": "Freeway", "tpke": "Turnpike", "plz": "Plaza", "aly": "Alley",
    "byp": "Bypass", "cswy": "Causeway", "est": "Estate", "ests": "Estates", "gdns": "Gardens", "grv": "Grove",
    "hbr": "Harbor", "holw": "Hollow", "is": "Island", "lk": "Lake", "mdws": "Meadows", "mnr": "Manor", "mtn": "Mountain",
    "orch": "Orchard", "pk": "Park", "rdg": "Ridge", "riv": "River", "shr": "Shore", "spg": "Spring", "spgs": "Springs",
    "sta": "Station", "vly": "Valley", "vlg": "Village", "vw": "View", "wy": "Way", "xrd": "Crossroad", "cty": "County",
    "tk": "Trunk", "rte": "Route", "rt": "Route",
}
#: Whole-component units (US): "Suite 100", "Unit 11", "# 109", "#K", "PMB 6098",
#: "3rd Floor", "PO Box 2821". A "#" followed by two or more tokens that is not a
#: unit word is a street with a stray marker ("#K 1510 HYLAND RD") and is NOT a unit.
US_UNIT_RX = re.compile(
    r"^(?:#\s*)?(?:(?:suite|ste|unit|apt|apartment|floor|flr|bldg|building|rm|room|office|ofc|pmb|lot|space|spc|dept)\b.*"
    r"|fl\s*\.?\s*\d+.*|\d+(?:st|nd|rd|th)\s+(?:floor|fl)\b.*|p\.?\s*o\.?\s*box\b.*|post\s+office\s+box\b.*)$"
    r"|^#\s*[A-Za-z0-9\-]+$",
    re.I,
)
US_UNIT_TAIL_RX = re.compile(r"\s+(?:suite|ste|unit|apt|apartment|floor|bldg|rm|room|pmb)\.?\s*#?\s*[A-Za-z0-9\-]+\s*$|\s+#\s*[A-Za-z0-9\-]+\s*$", re.I)
RX_CITY_OF = re.compile(r"^(?:town|city|village|borough|township|county)\s+of\s+", re.I)
DIRECTIONS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west"})

# --------------------------------------------------------------------------
# Addresses: India
# --------------------------------------------------------------------------

IN_STATE_CANON = [
    "Maharashtra", "Delhi", "Uttar Pradesh", "Karnataka", "Tamil Nadu", "Gujarat", "West Bengal", "Telangana",
    "Haryana", "Kerala", "Rajasthan", "Bihar", "Madhya Pradesh", "Andhra Pradesh", "Odisha", "Punjab",
    "Uttarakhand", "Himachal Pradesh", "Jharkhand", "Chhattisgarh", "Assam", "Goa", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Sikkim", "Tripura", "Arunachal Pradesh", "Jammu and Kashmir", "Chandigarh",
    "Puducherry", "Ladakh", "Dadra and Nagar Haveli", "Daman and Diu", "Andaman and Nicobar Islands", "Lakshadweep",
]
IN_STATES: dict[str, str] = {key(s): s for s in IN_STATE_CANON}
IN_STATES.update({
    "mh": "Maharashtra", "dl": "Delhi", "up": "Uttar Pradesh", "ka": "Karnataka", "tn": "Tamil Nadu", "gj": "Gujarat",
    "wb": "West Bengal", "tg": "Telangana", "ts": "Telangana", "hr": "Haryana", "kl": "Kerala", "rj": "Rajasthan",
    "br": "Bihar", "mp": "Madhya Pradesh", "ap": "Andhra Pradesh", "od": "Odisha", "or": "Odisha", "pb": "Punjab",
    "uk": "Uttarakhand", "ut": "Uttarakhand", "hp": "Himachal Pradesh", "jh": "Jharkhand", "cg": "Chhattisgarh",
    "ct": "Chhattisgarh", "as": "Assam", "ga": "Goa", "mn": "Manipur", "ml": "Meghalaya", "mz": "Mizoram",
    "nl": "Nagaland", "sk": "Sikkim", "tr": "Tripura", "ar": "Arunachal Pradesh", "jk": "Jammu and Kashmir",
    "ch": "Chandigarh", "py": "Puducherry", "la": "Ladakh", "dn": "Dadra and Nagar Haveli", "dd": "Daman and Diu",
    "an": "Andaman and Nicobar Islands", "ld": "Lakshadweep",
    "orissa": "Odisha", "keralam": "Kerala", "uttaranchal": "Uttarakhand", "pondicherry": "Puducherry",
    "tamilnadu": "Tamil Nadu", "tamil nadu": "Tamil Nadu", "chattisgarh": "Chhattisgarh", "chhatisgarh": "Chhattisgarh",
    "jammu kashmir": "Jammu and Kashmir", "jammu and kashmir": "Jammu and Kashmir", "j k": "Jammu and Kashmir",
    "new delhi": "Delhi", "nct of delhi": "Delhi", "delhi ncr": "Delhi", "andhra": "Andhra Pradesh",
    "telengana": "Telangana", "telangana state": "Telangana", "gujrat": "Gujarat", "haryana state": "Haryana",
    "bengal": "West Bengal", "west bengal state": "West Bengal", "madhya pradesh state": "Madhya Pradesh",
    "dadra nagar haveli": "Dadra and Nagar Haveli", "andaman nicobar": "Andaman and Nicobar Islands",
})
#: State names written in the state's own script (Source 2/3). The learned
#: component dictionary extends this list from ground-truth pairs.
IN_STATE_INDIC = {
    "महाराष्ट्र": "Maharashtra", "दिल्ली": "Delhi", "नई दिल्ली": "Delhi", "उत्तर प्रदेश": "Uttar Pradesh",
    "ಕರ್ನಾಟಕ": "Karnataka", "தமிழ்நாடு": "Tamil Nadu", "தமிழ் நாடு": "Tamil Nadu", "পশ্চিমবঙ্গ": "West Bengal",
    "ગુજરાત": "Gujarat", "తెలంగాణ": "Telangana", "हरियाणा": "Haryana", "राजस्थान": "Rajasthan",
    "കേരളം": "Kerala", "बिहार": "Bihar", "मध्य प्रदेश": "Madhya Pradesh", "ఆంధ్రప్రదేశ్": "Andhra Pradesh",
    "ఆంధ్ర ప్రదేశ్": "Andhra Pradesh", "ਪੰਜਾਬ": "Punjab", "ଓଡ଼ିଶା": "Odisha", "ଓଡିଶା": "Odisha",
    "उत्तराखंड": "Uttarakhand", "हिमाचल प्रदेश": "Himachal Pradesh", "झारखंड": "Jharkhand",
    "छत्तीसगढ़": "Chhattisgarh", "असम": "Assam", "गोवा": "Goa", "ਹਰਿਆਣਾ": "Haryana", "ਚੰਡੀਗੜ੍ਹ": "Chandigarh",
    "চণ্ডীগড়": "Chandigarh", "पंजाब": "Punjab", "গোয়া": "Goa",
}
IN_CITY_ALIAS: dict[str, str] = {
    "bangalore": "Bengaluru", "bombay": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai", "gurgaon": "Gurugram",
    "poona": "Pune", "baroda": "Vadodara", "trivandrum": "Thiruvananthapuram", "cochin": "Kochi", "mysore": "Mysuru",
    "mangalore": "Mangaluru", "belgaum": "Belagavi", "hubli": "Hubballi", "allahabad": "Prayagraj", "cawnpore": "Kanpur",
    "trichy": "Tiruchirappalli", "tiruchirapalli": "Tiruchirappalli", "vizag": "Visakhapatnam", "simla": "Shimla",
    "panjim": "Panaji", "secunderabad": "Secunderabad", "noida": "Noida", "navi mumbai": "Navi Mumbai",
    "bengaluru urban": "Bengaluru", "bangalore urban": "Bengaluru", "bangalore rural": "Bengaluru",
    "mumbai city": "Mumbai", "mumbai suburban": "Mumbai", "kolkata": "Kolkata", "hyderabad": "Hyderabad",
    "gautam buddha nagar": "Noida", "gautam budh nagar": "Noida", "thane": "Thane",
}
IN_STREET: dict[str, str] = {
    "rd": "Road", "st": "Street", "nr": "Near", "opp": "Opposite", "opp.": "Opposite", "bldg": "Building",
    "apts": "Apartments", "apt": "Apartment", "soc": "Society", "socy": "Society", "hsg": "Housing", "ind": "Industrial",
    "indl": "Industrial", "indus": "Industrial", "est": "Estate", "mkt": "Market", "ngr": "Nagar", "sec": "Sector",
    "sect": "Sector", "ph": "Phase", "flr": "Floor", "fl": "Floor", "ext": "Extension", "extn": "Extension",
    "encl": "Enclave", "col": "Colony", "cly": "Colony", "twp": "Township", "stn": "Station", "dist": "District",
    "distt": "District", "teh": "Tehsil", "tq": "Taluka", "tal": "Taluka", "vill": "Village", "vil": "Village",
    "gf": "Ground Floor", "ff": "First Floor", "sf": "Second Floor", "tf": "Third Floor", "ugf": "Upper Ground Floor",
    "lgf": "Lower Ground Floor", "bsmt": "Basement", "chs": "CHS", "co-op": "Cooperative", "coop": "Cooperative",
    "cplx": "Complex", "compl": "Complex", "cmplx": "Complex", "blk": "Block", "ln": "Lane", "cr": "Cross",
    "mn": "Main", "hosp": "Hospital", "univ": "University", "sch": "School", "off": "Off", "vlg": "Village",
    "gali": "Gali", "gl": "Gali", "mrg": "Marg", "jn": "Junction", "jct": "Junction", "cir": "Circle",
    "sq": "Square", "ave": "Avenue", "av": "Avenue", "blvd": "Boulevard", "dr": "Drive", "pkwy": "Parkway",
    "hwy": "Highway", "nh": "NH", "sh": "SH", "po": "Post", "p.o": "Post", "ps": "Police Station",
}
IN_LANDMARK = frozenset({"near", "opp", "opposite", "behind", "beside", "next", "adjacent", "above", "below", "infront", "front"})

# --------------------------------------------------------------------------
# Addresses: France
# --------------------------------------------------------------------------

FR_REGIONS = [
    "Auvergne-Rhone-Alpes", "Bourgogne-Franche-Comte", "Bretagne", "Centre-Val de Loire", "Corse", "Grand Est",
    "Hauts-de-France", "Ile-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire",
    "Provence-Alpes-Cote d'Azur", "Guadeloupe", "Martinique", "Guyane", "La Reunion", "Mayotte",
]
FR_REGION_KEY: dict[str, str] = {key(r): r for r in FR_REGIONS}
FR_REGION_KEY.update({"ile de france": "Ile-de-France", "paca": "Provence-Alpes-Cote d'Azur", "provence alpes cote d azur": "Provence-Alpes-Cote d'Azur",
                      "nouvelle aquitaine": "Nouvelle-Aquitaine", "hauts de france": "Hauts-de-France", "auvergne rhone alpes": "Auvergne-Rhone-Alpes",
                      "bourgogne franche comte": "Bourgogne-Franche-Comte", "centre val de loire": "Centre-Val de Loire", "pays de loire": "Pays de la Loire"})
_FR_DEPTS = {
    "Auvergne-Rhone-Alpes": ["Ain", "Allier", "Ardeche", "Cantal", "Drome", "Isere", "Loire", "Haute-Loire", "Puy-de-Dome", "Rhone", "Savoie", "Haute-Savoie", "Metropole de Lyon"],
    "Bourgogne-Franche-Comte": ["Cote-d'Or", "Doubs", "Jura", "Nievre", "Haute-Saone", "Saone-et-Loire", "Yonne", "Territoire de Belfort"],
    "Bretagne": ["Cotes-d'Armor", "Finistere", "Ille-et-Vilaine", "Morbihan"],
    "Centre-Val de Loire": ["Cher", "Eure-et-Loir", "Indre", "Indre-et-Loire", "Loir-et-Cher", "Loiret"],
    "Corse": ["Corse-du-Sud", "Haute-Corse"],
    "Grand Est": ["Ardennes", "Aube", "Marne", "Haute-Marne", "Meurthe-et-Moselle", "Meuse", "Moselle", "Bas-Rhin", "Haut-Rhin", "Vosges"],
    "Hauts-de-France": ["Aisne", "Nord", "Oise", "Pas-de-Calais", "Somme"],
    "Ile-de-France": ["Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise"],
    "Normandie": ["Calvados", "Eure", "Manche", "Orne", "Seine-Maritime"],
    "Nouvelle-Aquitaine": ["Charente", "Charente-Maritime", "Correze", "Creuse", "Dordogne", "Gironde", "Landes", "Lot-et-Garonne", "Pyrenees-Atlantiques", "Deux-Sevres", "Vienne", "Haute-Vienne"],
    "Occitanie": ["Ariege", "Aude", "Aveyron", "Gard", "Haute-Garonne", "Gers", "Herault", "Lot", "Lozere", "Hautes-Pyrenees", "Pyrenees-Orientales", "Tarn", "Tarn-et-Garonne"],
    "Pays de la Loire": ["Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendee"],
    "Provence-Alpes-Cote d'Azur": ["Alpes-de-Haute-Provence", "Hautes-Alpes", "Alpes-Maritimes", "Bouches-du-Rhone", "Var", "Vaucluse"],
}
FR_DEPT_TO_REGION: dict[str, str] = {key(d): r for r, ds in _FR_DEPTS.items() for d in ds}
FR_DEPT_TO_REGION.pop("paris", None)  # Paris is a city first; keep it as the city component
FR_STREET: dict[str, str] = {
    "r": "Rue", "bd": "Boulevard", "bld": "Boulevard", "boul": "Boulevard", "blvd": "Boulevard", "av": "Avenue", "ave": "Avenue",
    "all": "Allee", "alle": "Allee", "allée": "Allee", "allee": "Allee", "imp": "Impasse", "ch": "Chemin", "che": "Chemin",
    "chem": "Chemin", "rte": "Route", "pl": "Place", "sq": "Square", "crs": "Cours", "fg": "Faubourg", "fbg": "Faubourg",
    "res": "Residence", "rés": "Residence", "esp": "Esplanade", "prom": "Promenade",
    "sent": "Sentier", "qu": "Quai", "cite": "Cite", "cité": "Cite", "hlm": "HLM", "za": "ZA", "zi": "ZI", "zac": "ZAC",
    "blvd.": "Boulevard", "rpt": "Rond-Point", "rdpt": "Rond-Point", "mte": "Montee",
    "trav": "Traverse", "vla": "Villa", "chs": "Chaussee", "chaussée": "Chaussee",
    # dropped after review: "gal" (Général), "car", "pas", "lot" are ordinary words too
}
#: Street-type tokens that are only abbreviations at the START of a street
#: component ("R. de Dieppe"), never inside a name ("Avenue R. Salengro").
FR_STREET_LEAD_ONLY = frozenset({"r", "av", "all", "ch", "pl", "imp"})
FR_FLOOR_RX = re.compile(r"^\s*(?:\d+\s*(?:er|ere|eme|ème|e)\s+[eé]tage|[eé]tage\s+\d+|rdc|rez[- ]de[- ]chauss[eé]e|appt\.?\s*\S+|apt\.?\s*\S+|bat\.?\s*\S+|batiment\s+\S+|b[âa]t\s+\S+|cedex.*|bp\s*\d+|cs\s*\d+)\s*$", re.I)

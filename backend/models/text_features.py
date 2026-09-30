import re


LANGUAGE_HINTS = {
    "es": ["hola", "gracias", "que", "muy", "increíble", "estupendo", "bueno", "excelente"],
    "fr": ["bonjour", "merci", "très", "super", "incroyable", "magnifique", "bravo"],
    "de": ["hallo", "danke", "sehr", "super", "unglaublich", "glückwunsch", "toll"],
    "pt": ["olá", "obrigado", "muito", "incrível", "parabéns", "ótimo"],
    "it": ["ciao", "grazie", "molto", "incredibile", "complimenti", "bello"],
    "ar": ["مرحبا", "شكرا", "جيد", "رائع", "مبروك", "ممتاز"],
    "hi": ["नमस्ते", "धन्यवाद", "बहुत", "अच्छा", "बधाई", "शानदार"],
    "zh": ["你好", "谢谢", "非常", "太好了", "恭喜", "精彩"],
    "ja": ["こんにちは", "ありがとう", "すごい", "素晴らしい", "おめでとう"],
    "ru": ["привет", "спасибо", "очень", "замечательно", "поздравляю", "отлично"],
    "tr": ["merhaba", "teşekkürler", "çok", "harika", "tebrikler", "süper"],
    "nl": ["hallo", "dank je", "heel", "geweldig", "gefeliciteerd", "mooi"],
    "pl": ["cześć", "dziękuję", "bardzo", "wspaniale", "gratulacje", "świetnie"],
    "uk": ["привіт", "дякую", "дуже", "чудово", "вітаємо", "гарно"],
    "bn": [
        "ধন্যবাদ", "ভালো", "সুন্দর", "কেমন", "আছেন", "অনেক", "দাম", "কত", "কতো", "টাকা",
        "ভাই", "ভাইয়া", "ভাইয়া", "কিভাবে", "কীভাবে", "পাবো", "পাব", "অর্ডার", "করব", "করবো", "অভিনন্দন",
        "দারুণ", "অসাধারণ", "চমৎকার", "খুবই", "সেরা",
        "dhonnobad", "dhanyabad", "kemon", "achen", "bhalo", "valo", "onek", "shundor",
        "sundor", "dam", "koto", "bhai", "vai", "apnader", "taka", "shuvo", "shobai", "khub",
    ],
}

# ─────────────────────────────────────────────────────────────
# SENTIMENT LEXICONS (Expanded & Nuanced)
# ─────────────────────────────────────────────────────────────

STRONG_POSITIVE_WORDS = {
    "win", "wins", "winner", "winners", "winning", "won", "victory", "victories", "victorious",
    "champion", "champions", "championship", "triumph", "triumphant", "masterclass",
    "masterpiece", "spectacular", "phenomenal", "breathtaking", "unbelievable", "incredible",
    "outstanding", "extraordinary", "magnificent", "brilliant", "superb", "flawless",
    "perfection", "perfect", "legendary", "historic", "sensational", "marvelous", "stunning",
    "gorgeous", "delight", "delighted", "delightful", "overjoyed", "ecstatic", "thrilled",
    "thrilling", "excited", "exciting", "excitement", "proud", "proudly", "blessed", "grateful",
    "gratitude", "congratulations", "congrats", "celebrate", "celebrating", "celebration",
    "achievement", "achieved", "accomplishment", "milestone", "breakthrough", "inspire",
    "inspiring", "inspirational", "hero", "heroes", "heroic", "love", "loved", "loving",
    "awesome", "fantastic", "amazing", "wonderful", "fabulous", "splendid", "topnotch",
    "top-notch", "masterful", "sublime", "unmatched", "undefeated", "unbeatable", "dominant",
    "dominance", "mastery", "elite", "glory", "glorious", "pure class", "world-class",
    # Bengali Strong Positive
    "অভিনন্দন", "চমৎকার", "দারুণ", "অসাধারণ", "সেরা", "বিজয়", "জয়", "উদ্বোধন", "কৃতজ্ঞতা",
    "সাফল্য", "সফল", "অনবদ্য", "ধন্যবাদ", "শুভকামনা", "ভালোবাসা", "উচ্ছ্বসিত", "গর্বিত", "মাশাল্লাহ",
    "dhonnobad", "congrats", "osadharon", "darun", "seraa", "shubho",
}

MILD_POSITIVE_WORDS = {
    "good", "great", "nice", "fine", "happy", "happiness", "joy", "joyful", "pleasant",
    "pleased", "helpful", "useful", "benefit", "beneficial", "valuable", "effective",
    "efficient", "smart", "clever", "friendly", "kind", "kindness", "sweet", "cool",
    "solid", "steady", "promising", "favorable", "success", "successful", "progress",
    "improvement", "improve", "improved", "improving", "launch", "launched", "welcome",
    "appreciate", "appreciated", "cheers", "positive", "bright", "smooth", "smoothly",
    "clean", "neat", "fun", "funny", "enjoy", "enjoyed", "enjoying", "enjoyable",
    "satisfying", "satisfied", "recommend", "recommended", "worthwhile", "healthy",
    # Bengali Mild Positive
    "ভালো", "সুন্দর", "ভালোই", "উপকারী", "পছন্দ", "খুশি", "আনন্দ", "স্বাগতম", "প্রগতি",
    "উন্নতি", "আকর্ষণীয়", "কার্যকরী", "সহায়তা", "সন্তুষ্ট", "চমক", "দারুন",
    "valo", "bhalo", "shundor", "sundor", "khushi",
}

POSITIVE_WORDS = STRONG_POSITIVE_WORDS | MILD_POSITIVE_WORDS

STRONG_NEGATIVE_WORDS = {
    "disaster", "disastrous", "catastrophe", "catastrophic", "terrible", "horrible", "awful",
    "horrific", "dreadful", "abysmal", "tragic", "tragedy", "heartbroken", "heartbreaking",
    "devastating", "devastated", "furious", "outraged", "disgusted", "disgusting", "revolting",
    "repulsive", "abhorrent", "worthless", "hopeless", "ruined", "ruin", "destroyed",
    "destruction", "collapse", "collapsed", "nightmare", "crisis", "scandal", "atrocious",
    "appalling", "unbearable", "unacceptable", "intolerable", "painful", "agony", "miserable",
    "misery", "depressed", "depression", "despair", "furious", "enraged", "humiliated",
    "humiliating", "scammed", "betrayed", "betrayal", "cheated", "fraud", "scam",
    # Bengali Strong Negative
    "দুর্যোগ", "সর্বনাশ", "ভয়াবহ", "জঘন্য", "প্রতারণা", "প্রতারক", "জালিয়াতি", "চুরি",
    "বাজে", "ফালতু", "ঘৃণ্য", "লজ্জাজনক", "ধ্বংস", "হতাশাজনক", "ভয়ংকর",
    "faltu", "baje", "joghonno",
}

MILD_NEGATIVE_WORDS = {
    "bad", "sad", "unhappy", "angry", "anger", "frustrated", "frustrating", "frustration",
    "disappointed", "disappointing", "disappointment", "annoying", "annoyed", "annoyance",
    "boring", "bored", "slow", "delay", "delayed", "late", "broke", "broken", "fail",
    "failed", "failure", "failing", "poor", "poorly", "flawed", "mistake", "error",
    "wrong", "ugly", "waste", "wasted", "regret", "regrettable", "loss", "lost",
    "losing", "miss", "missed", "worried", "worry", "worrying", "concerned", "concern",
    "problem", "problems", "issue", "issues", "struggle", "struggling", "trouble",
    "difficult", "hard", "tough", "expensive", "costly", "confusing", "confused",
    "tiring", "tired", "exhausted", "exhausting", "awkward", "pity", "bummer",
    # Bengali Mild Negative
    "খারাপ", "সমস্যা", "দেরি", "বিরক্তিকর", "বিরক্ত", "ভুল", "ক্ষতি", "কঠিন", "দুঃখজনক",
    "ব্যর্থ", "ব্যর্থতা", "চিন্তিত", "কষ্ট", "বিপদ",
    "kharap", "shomossha", "kothin", "deri",
}

NEGATIVE_WORDS = STRONG_NEGATIVE_WORDS | MILD_NEGATIVE_WORDS

# ─────────────────────────────────────────────────────────────
# NEGATION, INTENSIFIERS & SARCASTIC MARKERS
# ─────────────────────────────────────────────────────────────

NEGATION_WORDS = {
    "not", "never", "no", "hardly", "barely", "scarcely", "without", "none",
    "cannot", "can't", "cant", "won't", "wont", "don't", "dont", "didn't",
    "didnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't",
    "werent", "haven't", "havent", "hasn't", "hasnt", "hadn't", "hadnt",
    "neither", "nor", "nothing", "nowhere", "noway", "no way",
    # Bengali Negation
    "না", "নাই", "নেই", "নয়", "হবে না", "পারব না", "পারবে না", "কখনো না", "কখনই না",
    "na", "nai", "nei", "noy",
}

INTENSIFIERS = {
    "very", "extremely", "incredibly", "insanely", "unbelievably", "totally",
    "absolutely", "completely", "so", "really", "super", "hugely", "deeply",
    "truly", "wildly", "immensely", "exceptionally", "enormously",
    # Bengali Intensifiers
    "খুব", "খুবই", "অনেক", "অত্যন্ত", "একেবারে", "পুরোপুরি", "দারুণভাবে", "ভীষণ", "মারাত্মক",
    "khub", "khubi", "onek", "beshi",
}

DIMINISHERS = {
    "slightly", "somewhat", "a bit", "a little", "mildly", "barely", "hardly",
    "kind of", "kinda", "sort of", "sorta", "partially",
    # Bengali Diminishers
    "একটু", "সামান্য", "কিছুটা", "কম", "হালকা",
    "ektu", "kichuta",
}

SARCASM_MARKERS = [
    "yeah right", "as if", "sure thing", "totally normal", "just what i needed",
    "oh great", "big surprise", "thanks a lot for nothing", "slow clap", "/s",
    "🙃", "🙄", "what a shock", "groundbreaking discovery",
    "বাহ কি দারুণ", "অনেক উপকার করলেন",
]

# ─────────────────────────────────────────────────────────────
# TOXICITY CATEGORIES & TARGET DISCRIMINATION
# ─────────────────────────────────────────────────────────────

# Direct personal insults directed at individuals or groups
PERSONAL_INSULTS = {
    "idiot", "idiots", "moron", "morons", "dumbass", "dumbasses", "clown", "clowns",
    "loser", "losers", "pathetic", "worthless", "scumbag", "scumbags", "piece of shit",
    "pos", "creep", "creeps", "freak", "freaks", "bastard", "bastards", "bitch", "bitches",
    "asshole", "assholes", "dickhead", "dickheads", "dipshit", "douchebag", "douchebags",
    "incompetent fool", "you suck", "you all suck", "disgusting pig", "ugly pig",
    "shut the fuck up", "stfu", "gtfo", "get lost", "waste of space", "waste of oxygen",
    "eat shit", "retard", "retarded", "jackass", "scum", "trash person", "human garbage",
    "get a life", "horrible person", "shut up", "toxic person", "shut your mouth",
    # Bengali Insults
    "গাধা", "পাগল", "বেকুব", "বেয়াদব", "বেয়াদব", "হারামি", "কুত্তা", "খানকির", "শুয়োরের",
    "শুয়োর", "শালা", "শয়তান", "শয়তান", "চোর", "বাটপার", "চিটার", "ধান্দাবাজ", "বদমাইশ",
    "অপদার্থ", "নির্লজ্জ", "বোকা",
    "gadha", "pagol", "harami", "kutta", "khankir", "shala", "shoytan", "batpar", "badmaish",
}

# Violent threats and incitement of self-harm
VIOLENT_THREATS = {
    "kill yourself", "kys", "go die", "i will kill you", "want to kill you", "die in a hole",
    "hope you die", "murder you", "slit your", "shoot you", "strangle you", "beat you up",
    "beat the shit out of", "doxx you", "burn in hell", "die of cancer", "commit suicide",
    "drink bleach", "jump off a cliff", "slit my wrists", "kill myself", "end my life",
    # Bengali Violent Threats
    "মেরে ফেলব", "মেরে ফেলবো", "খুন করব", "খুন করবো", "মর", "মর তুই", "জাহান্নামে যা",
    "গলা কেটে", "গুলি করে মারব", "পিটিয়ে মারব",
    "mere felbo", "khun korbo", "mor tui",
}

# Vulgar profanities and slurs
PROFANITIES = {
    "fuck", "fucking", "fucked", "motherfucker", "motherfucking", "cunt", "cunts",
    "shit", "bullshit", "horseshit", "piss off", "dick", "cock", "whore", "slut",
    # Bengali Profanities
    "কুত্তার বাচ্চা", "শুয়োরের বাচ্চা", "খানকি", "খানকির পোলা", "হারামির বাচ্চা", "মাগির পোলা",
}

# Scams & deceptive spam patterns
SCAM_PATTERNS = [
    r"\b(?:guaranteed|double\s+your)\s+(?:crypto|bitcoin|btc|money|investment)\b",
    r"\b(?:whatsapp|telegram)\s+(?:me|inbox|dm)\s+(?:for\s+invest|to\s+earn)\b",
    r"\b(?:100%|free)\s+(?:payout|returns|giftcard|airdrop)\s+(?:click|dm)\b",
]

# Sensitive safety topics requiring review/flagging
SENSITIVE_TOPICS = {
    "suicide", "self-harm", "drugs", "terrorism", "violence", "weapon", "porn",
    "gambling", "medication dosage", "legal advice", "medical diagnosis",
}

# Backwards compatible export
TOXIC_WORDS = PERSONAL_INSULTS | VIOLENT_THREATS | {"scam", "fraud"}

# ─────────────────────────────────────────────────────────────
# CONTEXT DISCRIMINATION PATTERNS (Anti-False-Positive)
# ─────────────────────────────────────────────────────────────

# Harmless usages of the word "hate" (disliking conditions, objects, or delays — not people)
BENIGN_HATE_RE = re.compile(
    r"\bhate\s+(?:when|the\b|this\b|that\b|traffic|rain|mondays|weather|delays?|cold|waiting|homework|chores|bugs?|cleaning|mondays|waking\s+up|headaches?|flying|cooking|spicy\s+food)\b",
    re.IGNORECASE,
)

# Harmless usages of the word "trash" (literal garbage disposal or sports context)
BENIGN_TRASH_RE = re.compile(
    r"\b(?:take\s+out\s+the|throw\s+(?:in|into|away\s+in)\s+the|empty\s+the|dumpster|bin|can|pile)\s+trash\b|\btrash\s+(?:can|bin|bag|bags|collector|truck)\b",
    re.IGNORECASE,
)

# Harmless usages of the word "kill" (idiomatic expressions)
BENIGN_KILL_RE = re.compile(
    r"\b(?:killing\s+it|killed\s+it|kill\s+time|killing\s+time|kill\s+a\s+bug|kill\s+the\s+lights|killing\s+me\s+with\s+laughter|buzzkill)\b",
    re.IGNORECASE,
)

# Harmless usages of words like "stupid" directed at situations/tools rather than people
BENIGN_STUPID_RE = re.compile(
    r"\bstupid\s+(?:bug|code|traffic|weather|error|glitch|machine|computer|laptop|phone|alarm|car|bus|train|flight|delay|system|rule|thing|idea|game)\b",
    re.IGNORECASE,
)

# Harmless usages of words like "sucks" directed at situations/circumstances
BENIGN_SUCK_RE = re.compile(
    r"\b(?:traffic|delays?|weather|rain|mondays?|chores?|waiting|bugs?|commute)\s+sucks?\b",
    re.IGNORECASE,
)

CONTRASTIVE_CONJUNCTIONS = {
    "but", "however", "although", "though", "yet", "nevertheless", "nonetheless", "still",
    "কিন্তু", "তবে", "অথচ", "তা সত্ত্বেও",
}

POSITIVE_SLANG = {
    "killing it", "killed it", "slaying", "slayed", "goat", "fire", "banger",
    "top tier", "game changer", "masterclass", "pure gold",
}


def is_question_intent(text):
    """Check if the text represents a question, inquiry, or request for advice/recommendations."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    lower = stripped.lower()
    q_markers = [
        "what", "how", "which", "where", "why", "who", "when",
        "should i", "could anyone", "can someone", "does anyone", "any tips", "any advice",
        "any recommendations", "what's the best", "what is the best", "recommend me", "thoughts on",
        "কীভাবে", "কিভাবে", "কোনটা", "কেন", "কখন", "কোথায়", "কোথায়", "পরামর্শ", "উপদেশ", "কেমন হবে"
    ]
    return any(lower.startswith(m) or f" {m} " in lower for m in q_markers)



def tokens(text):
    """Extract word tokens from text while preserving case-insensitive alphanumeric and Bengali characters."""
    return re.findall(r"[a-zA-Z0-9\u0980-\u09FF#']+", text.lower())


def hashtags(text):
    """Extract hashtags from text."""
    return re.findall(r"#[a-zA-Z0-9\u0980-\u09FF_]+", text)


def count_matches(text, words):
    """Count token or phrase occurrences in text using word boundaries where appropriate."""
    lower = text.lower()
    token_set = set(tokens(lower))
    count = 0
    for word in words:
        if " " in word:
            if word in lower:
                count += 1
        else:
            if word in token_set:
                count += 1
    return count


def feature_vector(text, sentiment="neutral"):
    """Compute structured feature vector for social posts."""
    words = tokens(text)
    char_count = len(text)
    word_count = len(words)
    hash_count = len(hashtags(text))
    pos = count_matches(text, POSITIVE_WORDS)
    neg = count_matches(text, NEGATIVE_WORDS)
    sentiment_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(sentiment, 0.0)
    return [word_count, char_count, hash_count, pos, neg, sentiment_score]


def detect_language(text):
    """Heuristic language detection based on distinctive vocabulary, Bengali script, and character sets."""
    if not text:
        return "en"
    # 1. Deterministic check for Bengali script (\u0980-\u09FF)
    if any("\u0980" <= char <= "\u09FF" for char in text):
        return "bn"
    lower = text.lower()
    token_set = set(tokens(lower))
    # 2. Check language hint vocabulary (including transliterated Banglish)
    for code, hints in LANGUAGE_HINTS.items():
        if any((hint in token_set) or ((" " in hint) and (hint in lower)) for hint in hints):
            return code
    if any(char in lower for char in "éàèùêîôûçœæïüöäß"):
        return "fr"
    if any(ord(char) > 127 for char in lower):
        return "multilingual"
    return "en"


def extract_dynamic_topics(text):
    """Extract meaningful domain topics and entities from text rather than static defaults or sentence starters."""
    if not text:
        return ["general"]

    # 1. Look for explicit hashtags first
    tags = [tag.strip("#") for tag in hashtags(text)]
    if tags:
        clean_tags = [t for t in tags if len(t) > 2]
        if clean_tags:
            return clean_tags[:4]

    # Sentence starter / conversational filler words that might be capitalized
    common_title_words = {
        "The", "A", "An", "This", "That", "These", "Those", "In", "On", "At", "For", "With",
        "After", "Before", "I", "We", "You", "They", "He", "She", "It", "My", "Our", "Your",
        "Finally", "Today", "Yesterday", "Tomorrow", "Just", "So", "Well", "Now", "Here",
        "There", "Please", "Let", "Another", "Some", "Any", "If", "When", "While", "Someone",
        "Everyone", "Awesome", "Great", "Amazing", "Huge", "Good", "Bad", "New", "Old",
    }

    # 2. Extract multi-word capitalized proper nouns / brand names (e.g. "Desert Vipers", "DP World ILT20")
    multi_entities = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+\b", text)
    valid_multi = []
    for ent in multi_entities:
        words = ent.split()
        # If the first word is a common title starter, strip it if the rest is capitalized
        if words[0] in common_title_words and len(words) > 1:
            ent = " ".join(words[1:])
        if ent and ent not in valid_multi and len(ent) > 2:
            valid_multi.append(ent)

    if valid_multi:
        return valid_multi[:4]

    # 3. Check domain compound phrases
    lower = text.lower()
    compound_candidates = [
        "software engineering", "machine learning", "data science", "web development",
        "remote work", "work from home", "product launch", "ai dashboard", "final over",
        "puppy training", "homemade biryani", "flight delay", "lost luggage", "morning coffee",
        "road trip", "desert vipers", "customer service", "open source", "job search",
        "interview prep", "weekend vibes", "traffic jam", "car maintenance",
        # Bengali compounds
        "নতুন প্রজেক্ট", "শীতকালীন কালেকশন", "ম্যাচ জয়", "সফটওয়্যার ইঞ্জিনিয়ারিং", "ঘরের রান্না",
    ]
    matched_compounds = [c for c in compound_candidates if c in lower]
    if matched_compounds:
        return matched_compounds[:3]

    # 4. Extract meaningful nouns and phrases
    stopwords = {
        "the", "this", "that", "these", "those", "and", "or", "but", "for", "with",
        "about", "from", "into", "over", "after", "before", "just", "what", "when",
        "where", "how", "why", "who", "which", "our", "your", "their", "its", "it",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "will", "would", "shall", "should",
        "may", "might", "must", "very", "really", "much", "more", "most", "some",
        "any", "all", "such", "than", "too", "also", "then", "there", "here",
        "finally", "today", "yesterday", "first", "second", "months", "years", "hours",
        "days", "ever", "every", "got", "get", "getting", "made", "make", "went", "come",
        # Bengali stopwords
        "এই", "সেই", "এবং", "বা", "কিন্তু", "জন্য", "সাথে", "হতে", "থেকে", "করে",
        "হচ্ছে", "ছিল", "আছে", "হবে", "একটি", "একটা", "কোন", "কী", "কেন", "কিভাবে",
        "কীভাবে", "কখন", "কোথায়", "আমার", "আমাদের", "তোমার", "তোমাদের", "তার",
        "তাদের", "সব", "খুব", "একটু", "শুধু", "আজ", "কাল", "এখন", "তখন", "যদি",
        "আজকের", "দিনটি", "সত্যিই", "নতুন", "করলাম", "হলো", "গেছে", "দিয়ে",
    }

    meaningful_words = [t for t in tokens(text) if len(t) > 3 and t not in stopwords and t not in POSITIVE_WORDS and t not in NEGATIVE_WORDS]
    if meaningful_words:
        seen = []
        for w in meaningful_words:
            if w not in seen:
                seen.append(w)
            if len(seen) >= 3:
                break
        return seen

    return ["social update"]


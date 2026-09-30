import base64
import io
import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from PIL import Image, ImageStat, ImageFilter

logger = logging.getLogger("chatbot.vision")


class LocalVisionModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.source = "blip-image-captioning"
        self._load_attempted = False

    def _ensure_loaded(self):
        """Lazy-load the BLIP model on first call to optimize startup time."""
        if os.environ.get("RENDER") or os.environ.get("DISABLE_BLIP") or os.environ.get("LOW_MEMORY"):
            return False
        if self._load_attempted:
            return self.model is not None

        self._load_attempted = True
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_id = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
            logger.info("Local BLIP vision model loaded successfully.")
            return True
        except Exception as exc:
            logger.warning("Could not load local BLIP model: %s", exc)
            self.processor = None
            self.model = None
            return False

    def _try_gemini_vision(self, image_bytes):
        """Query Google Gemini 2.5 Flash multimodal vision API if GEMINI_API_KEY is configured."""
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not gemini_key:
            return None
        try:
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            encoded = base64.b64encode(image_bytes).decode("ascii")
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": (
                                "Describe what is happening in this image in 1 clear, natural sentence. "
                                "State the primary subjects, their actions, and the setting or background specifically."
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded,
                            }
                        },
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": 65,
                    "temperature": 0.2,
                },
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as res:
                data = json.loads(res.read().decode("utf-8"))
            candidates = data.get("candidates") or []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    caption = text.replace("\n", " ").strip()
                    if not caption.endswith((".", "!", "?")):
                        caption += "."
                    logger.info("Gemini vision produced caption: %s", caption)
                    return caption
        except Exception as exc:
            logger.warning("Gemini vision captioning failed: %s", exc)
        return None

    def _try_hf_serverless_blip(self, image_bytes):
        """Query Hugging Face's free serverless inference API for Salesforce/blip-image-captioning-base."""
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            return None

        endpoints = [
            "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base",
            "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
        ]
        for url in endpoints:
            try:
                req = urllib.request.Request(
                    url,
                    data=image_bytes,
                    headers={
                        "Authorization": f"Bearer {hf_token}",
                        "Content-Type": "application/octet-stream",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=12) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    if isinstance(data, list) and data and "generated_text" in data[0]:
                        raw = data[0]["generated_text"].strip()
                        if raw:
                            caption = raw[0].upper() + raw[1:]
                            if not caption.endswith((".", "!", "?")):
                                caption += "."
                            logger.info("HF serverless BLIP produced caption: %s", caption)
                            return caption
            except Exception as exc:
                logger.warning("HF Serverless BLIP endpoint (%s) failed: %s", url, exc)
        return None

    def _extract_semantic_keywords(self, filename="", user_prompt=""):
        """Extract high-confidence domain tags from filename and optional prompt."""
        text = f"{filename} {user_prompt}".lower()
        cleaned = re.sub(r"[_\-\.\d]", " ", text)
        words = set(cleaned.split())

        categories = {
            "mountain": {"mountain", "mountains", "summit", "peak", "peaks", "cliff", "ridge", "hike", "hiking", "hiker", "trail", "climbing", "climb", "alps", "altitude", "overlook", "elevation", "hills"},
            "pet": {"cat", "cats", "kitten", "kitty", "dog", "dogs", "puppy", "pup", "pet", "pets", "bird", "animal", "hamster", "bunny", "rabbit", "canine", "feline"},
            "food": {"food", "dish", "meal", "coffee", "cake", "pizza", "burger", "dinner", "lunch", "breakfast", "pasta", "tea", "snack", "recipe", "restaurant", "dessert", "biryani", "bakery"},
            "people": {"portrait", "selfie", "me", "friend", "friends", "brother", "sister", "mom", "dad", "family", "group", "team", "person", "smile", "smiling", "boy", "girl", "man", "woman", "couple"},
            "nature": {"flower", "flowers", "rose", "garden", "nature", "tree", "trees", "plant", "plants", "forest", "park", "bloom", "foliage", "botanical"},
            "coastal": {"beach", "sea", "ocean", "lake", "river", "boat", "waterfront", "island", "coast", "swimming", "sand", "waves", "water"},
            "sunset": {"sunset", "sunrise", "dusk", "dawn", "goldenhour", "golden", "evening", "twilight"},
            "celebration": {"birthday", "wedding", "anniversary", "party", "celebration", "ceremony", "graduation", "festival", "eid", "puja", "holiday"},
            "workspace": {"laptop", "desk", "office", "code", "coding", "setup", "workspace", "computer", "macbook", "study", "work"},
            "fitness": {"gym", "workout", "fitness", "run", "running", "sport", "football", "cricket", "match", "exercise", "training"},
            "vehicle": {"car", "bike", "motorcycle", "ride", "drive", "roadtrip", "vehicle", "automobile"},
        }

        detected = []
        for cat, kw_set in categories.items():
            if words.intersection(kw_set):
                detected.append(cat)
        return detected

    def _pure_python_describe(self, img, filename="", user_prompt=""):
        """Analyze image geometry, colorimetry, skin tone distribution, and semantic cues."""
        w, h = img.size
        aspect = w / h
        if aspect < 0.65:
            orientation = "vertical mobile portrait"
        elif aspect < 0.90:
            orientation = "portrait"
        elif aspect <= 1.15:
            orientation = "square"
        elif aspect <= 1.75:
            orientation = "landscape"
        else:
            orientation = "panoramic landscape"

        # Fast 120x120 thumbnail pixel analysis
        thumb = img.resize((120, 120)).convert("RGB")
        pixels = list(thumb.getdata())
        n = len(pixels)

        # 1. Skin tone clustering via YCbCr (Chai & Ngan standard model)
        ycbcr = thumb.convert("YCbCr")
        y_data = list(ycbcr.getdata())
        skin_total = 0
        skin_center = 0
        skin_rows = []
        for idx, (y, cb, cr) in enumerate(y_data):
            if 77 <= cb <= 127 and 133 <= cr <= 173 and y > 40:
                skin_total += 1
                row = idx // 120
                col = idx % 120
                skin_rows.append(row)
                if 25 <= row <= 95 and 25 <= col <= 95:
                    skin_center += 1

        skin_ratio = skin_total / n
        center_skin_ratio = skin_center / (70 * 70)
        has_face_portrait = center_skin_ratio > 0.12 or (skin_ratio > 0.10 and len(skin_rows) > 0 and (sum(skin_rows) / len(skin_rows)) < 65)
        has_group = skin_ratio > 0.22 and center_skin_ratio <= 0.18

        # 2. Color channel dominance
        green_dom = sum(1 for r, g, b in pixels if g > r + 15 and g > b + 15) / n
        blue_dom = sum(1 for r, g, b in pixels if b > r + 15 and b > g + 10) / n
        upper_blue = sum(1 for idx, (r, g, b) in enumerate(pixels) if idx < n // 2 and b > r + 15 and b > g) / (n // 2)
        lower_blue = sum(1 for idx, (r, g, b) in enumerate(pixels) if idx >= n // 2 and b > r + 15 and b > g) / (n // 2)
        warm_dom = sum(1 for r, g, b in pixels if r > 160 and g < 150 and b < 110) / n
        bright_dom = sum(1 for r, g, b in pixels if (r + g + b) / 3 > 210) / n
        dark_dom = sum(1 for r, g, b in pixels if (r + g + b) / 3 < 55) / n

        # 3. Semantic keyword extraction from filename / prompt
        tags = self._extract_semantic_keywords(filename=filename, user_prompt=user_prompt)

        # Compound semantic combinations first
        if "mountain" in tags and "pet" in tags and "people" in tags:
            return f"A scenic {orientation} photograph capturing an outdoor mountain adventure with a hiking companion and pet."
        if "mountain" in tags and "people" in tags:
            return f"An inspiring {orientation} outdoor photograph capturing a hiker exploring scenic mountain peaks."
        if "mountain" in tags and "pet" in tags:
            return f"A captivating {orientation} outdoor capture featuring an adventurous pet on a mountain trail."
        if "mountain" in tags:
            return f"A majestic {orientation} landscape photograph showcasing rugged mountain peaks and vast open skies."
        if "coastal" in tags and "sunset" in tags:
            return f"A breathtaking {orientation} sunset capture with warm golden reflections over the coastal waterfront."
        if "coastal" in tags and "people" in tags:
            return f"A vibrant {orientation} coastal capture featuring people enjoying the sunny beach and ocean atmosphere."
        if "food" in tags and "people" in tags:
            return f"A delightful {orientation} culinary capture featuring people sharing a delicious meal together."
        if "pet" in tags and "people" in tags:
            return f"A heartwarming {orientation} photograph capturing a sweet moment with a loyal pet companion."

        # Single semantic tags
        if "sunset" in tags:
            return f"A stunning {orientation} photograph capturing golden-hour sunset warmth with radiant skies."
        if "nature" in tags:
            return f"A refreshing {orientation} nature photograph highlighting lush greenery, foliage, and tranquil outdoor scenery."
        if "coastal" in tags:
            return f"A scenic {orientation} coastal capture highlighting open blue skies and waterfront surroundings."
        if "celebration" in tags:
            return f"A festive {orientation} photograph capturing a celebratory moment with friends and family."
        if "pet" in tags:
            env_txt = "in a cozy indoor setting" if dark_dom > 0.2 or bright_dom > 0.3 else "in a bright natural space"
            return f"A heartwarming {orientation} photograph featuring an adorable pet companion {env_txt}."
        if "workspace" in tags:
            return f"A modern {orientation} workspace photograph highlighting a productive tech setup and creative environment."
        if "fitness" in tags:
            return f"An energetic {orientation} lifestyle capture highlighting active fitness, training, or sports movement."
        if "vehicle" in tags:
            return f"A sharp {orientation} photograph showcasing a stylish vehicle with great framing and road appeal."
        if "food" in tags:
            return f"An appetizing {orientation} culinary photograph showcasing a delicious meal in an inviting dining setting."

        # Pixel heuristics
        if has_group or ("people" in tags and skin_ratio > 0.18):
            return f"A cheerful {orientation} photograph capturing people gathered together in high spirits."
        if has_face_portrait or ("people" in tags and skin_ratio > 0.08):
            lighting = "soft natural lighting" if bright_dom > 0.25 else "warm, atmospheric lighting" if dark_dom > 0.2 else "balanced indoor lighting"
            return f"A portrait photograph featuring a person smiling with {lighting}."
        if green_dom > 0.25:
            return f"A refreshing {orientation} nature photograph highlighting lush greenery, foliage, and tranquil outdoor scenery."
        if lower_blue > 0.25 and upper_blue > 0.2:
            return f"A scenic {orientation} coastal capture highlighting open blue skies and waterfront surroundings."
        if warm_dom > 0.25 and upper_blue < 0.15:
            return f"A stunning {orientation} photograph capturing golden-hour sunset warmth with radiant skies."
        if warm_dom > 0.18 and not has_face_portrait and dark_dom < 0.4:
            return f"An appetizing {orientation} culinary photograph showcasing a delicious meal in an inviting dining setting."
        if upper_blue > 0.35:
            return f"A bright {orientation} outdoor capture showcasing clear open skies and scenic background details."
        if dark_dom > 0.50:
            return f"A dramatic {orientation} night-time photograph featuring evening ambience and artistic lighting."
        if bright_dom > 0.60:
            return f"A clean, minimalist {orientation} photograph with bright illumination and modern composition."
        if warm_dom > 0.20:
            return f"A vibrant {orientation} visual composition featuring warm tones and distinct foreground focus."

        # Default balanced description
        stat = ImageStat.Stat(thumb)
        stddev = sum(stat.stddev[:3]) / 3
        if stddev > 50:
            return f"A colorful, visually engaging {orientation} photograph with rich textures and detailed framing."
        return f"A thoughtfully composed {orientation} visual image featuring focused subjects with balanced lighting."

    def describe_image(self, image_bytes, filename="", user_prompt=""):
        """Analyze the image and return a clear description of what is happening."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to open image for vision captioning: %s", exc)
            return "An uploaded photograph with distinct visual subjects."

        # 1. Try local BLIP AI captioning if loaded (localhost with PyTorch)
        if self._ensure_loaded():
            try:
                inputs = self.processor(img, return_tensors="pt")
                output = self.model.generate(**inputs, max_new_tokens=45)
                raw_caption = self.processor.decode(output[0], skip_special_tokens=True).strip()
                if raw_caption:
                    caption = raw_caption[0].upper() + raw_caption[1:]
                    if not caption.endswith((".", "!", "?")):
                        caption += "."
                    return caption
            except Exception as exc:
                logger.warning("BLIP generation failed: %s", exc)

        # 2. Try Gemini 2.5 Flash Multimodal Vision if key is configured (0 MB RAM, free, state-of-the-art)
        gemini_caption = self._try_gemini_vision(image_bytes)
        if gemini_caption:
            return gemini_caption

        # 3. Try Hugging Face Serverless BLIP inference if HF_TOKEN is configured
        hf_caption = self._try_hf_serverless_blip(image_bytes)
        if hf_caption:
            return hf_caption

        # 4. Advanced pure-Python visual analysis (Pillow pixel analysis + semantic tags)
        try:
            return self._pure_python_describe(img, filename=filename, user_prompt=user_prompt)
        except Exception as exc:
            logger.warning("Pure Python image description failed: %s", exc)
            w, h = img.size
            orientation = "landscape" if w > h else "portrait" if h > w else "square"
            return f"A {orientation} photograph featuring focused visual subjects in a well-balanced scene."

    def generate_comment(self, description, tone="friendly", platform="facebook", language=None):
        """Generate an ultra-specific, engaging social media comment tailored to the image description."""
        tone = (tone or "friendly").lower()
        if tone not in {"friendly", "enthusiastic", "professional"}:
            tone = "friendly"
        d_low = description.lower()

        # Check for Bengali language
        is_bangla = (language == "bn") or any("\u0980" <= c <= "\u09FF" for c in description)

        # ── ENTITY DETECTION ──
        has_pet = any(w in d_low for w in [
            "dog", "pup", "puppy", "dogs", "cat", "cats", "kitten", "kitty", "pet", "pets",
            "canine", "feline", "animal", "paws", "furry", "কুকুর", "বিড়াল", "বিড়াল", "পোষা প্রাণী"
        ])
        has_mountain = any(w in d_low for w in [
            "mountain", "mountains", "summit", "peak", "peaks", "cliff", "ridge", "hike", "hiking",
            "hiker", "trail", "climbing", "climb", "altitude", "overlook", "elevation", "alps", "rocky", "পাহাড়", "পর্বত", "চূড়া", "চূড়া"
        ])
        has_person = any(w in d_low for w in [
            "woman", "man", "girl", "boy", "person", "people", "someone", "friend", "friends",
            "hiker", "climber", "traveler", "couple", "smiling", "portrait", "sitting", "standing",
            "holding", "group", "crowd", "gathered", "নারী", "মহিলা", "পুরুষ", "মানুষ", "বন্ধু", "বন্ধুরা"
        ])
        has_beach = any(w in d_low for w in [
            "beach", "sea", "ocean", "coastal", "coast", "shore", "waterfront", "waves", "surf",
            "sand", "lake", "swimming", "সমুদ্র", "সৈকত", "নদী", "পানি"
        ])
        has_sunset = any(w in d_low for w in [
            "sunset", "sunrise", "golden-hour", "golden hour", "twilight", "dusk", "dawn", "sun setting", "skies", "সূর্যাস্ত", "সূর্যোদয়", "গোধূলি"
        ])
        has_food = any(w in d_low for w in [
            "food", "meal", "coffee", "cake", "dish", "plate", "dining", "dinner", "lunch", "breakfast",
            "pasta", "pizza", "burger", "dessert", "recipe", "delicious", "tasty", "culinary", "baking", "bread", "খাবার", "রান্না", "চা", "কফি"
        ])
        has_nature = any(w in d_low for w in [
            "flower", "flowers", "bloom", "blossom", "rose", "garden", "tree", "trees", "forest",
            "plants", "greenery", "foliage", "botanical", "woodland", "ফুল", "বাগান", "গাছ", "প্রকৃতি"
        ])
        has_celebration = any(w in d_low for w in [
            "birthday", "wedding", "anniversary", "party", "celebrating", "celebration", "ceremony",
            "graduation", "achievement", "trophy", "cheers", "festival", "festive", "জন্মদিন", "বিয়ে", "বার্ষিকী", "উৎসবে", "উৎসব"
        ])
        has_fitness = any(w in d_low for w in [
            "gym", "workout", "fitness", "training", "exercise", "running", "runner", "marathon",
            "athlete", "sports", "cricket", "football", "yoga", "cycling", "ব্যায়াম", "দৌড়", "খেলা"
        ])
        has_workspace = any(w in d_low for w in [
            "workspace", "desk", "laptop", "coding", "code", "office", "computer", "setup",
            "programmer", "developer", "working", "macbook", "অফিস", "ডেস্ক", "ল্যাপটপ"
        ])
        has_vehicle = any(w in d_low for w in [
            "car", "cars", "bike", "motorcycle", "vehicle", "automobile", "drive", "driving", "roadtrip", "ride", "গাড়ি", "গাড়ি", "বাইক"
        ])

        # ── BANGLA / BENGALI SPECIFIC RESPONSES ──
        if is_bangla:
            if has_person and has_pet and has_mountain:
                return "পাহাড়ের চূড়ায় প্রিয় কুকুরের সাথে অসাধারণ একটি অ্যাডভেঞ্চার! এই দৃশ্য এবং আপনাদের সুন্দর বন্ধন সত্যি মন ছুঁয়ে যায়। 🐾🏔️✨"
            if has_person and has_mountain:
                return "পাহাড়ের চূড়ায় দাঁড়িয়ে প্রকৃতির রূপ উপভোগ করার আনন্দই আলাদা! দারুণ একটি ভ্রমণ মুহূর্ত। 🏔️🙌✨"
            if has_pet and has_mountain:
                return "পাহাড়ের বুকে ছোট্ট অভিযাত্রী! দৃশ্যটি যেমন সুন্দর, পোষা বন্ধুটিকেও ততটাই আদুরে লাগছে। 🐾⛰️❤️"
            if has_person and has_pet:
                return "কী মিষ্টি এবং ভালোবাসায় ভরা একটি মুহূর্ত! আপনার বিশ্বস্ত চারপেয়ে বন্ধুর সাথে বন্ধন সত্যি চমৎকার। 🐶❤️"
            if has_pet:
                return "কী অপূর্ব এবং আদুরে একটি ছবি! মিষ্টি মুখখানা দেখে মনটা ভরে গেল। 🐾✨"
            if has_mountain:
                return "পাহাড়ের এই নৈসর্গিক রূপ সত্যিই অসাধারণ! প্রকৃতির বিশালতা চোখে পড়ার মতো। 🏔️🌿"
            if has_sunset and (has_beach or "water" in d_low):
                return "সাগরের বুকে মনোমুগ্ধকর সূর্যাস্ত! আকাশের রঙ আর পানির ঢেউয়ের প্রতিফলন সত্যিই অতুলনীয়। 🌅🌊✨"
            if has_sunset:
                return "সূর্যাস্তের এই সোনালী আভা অসাধারণ লাগছে! প্রকৃতির অপার সৌন্দর্য ক্যামেরায় সুন্দরভাবে ফুটে উঠেছে। 🌅💛"
            if has_beach:
                return "সাগরের স্নিগ্ধ রূপ এবং মনোরম পরিবেশ! অসাধারণ একটি উপকূলীয় মুহূর্ত। 🌊💙"
            if has_food:
                return "খাবারটি দেখতে অত্যন্ত সুস্বাদু ও লোভনীয় লাগছে! উপস্থাপনা সত্যিই অতুলনীয়। 🍽️😋"
            if has_celebration:
                return "বিশেষ এই অর্জনে অনেক অনেক শুভকামনা ও অভিনন্দন! সুন্দর মুহূর্তগুলো চিরস্মরণীয় হয়ে থাকুক। 🎉✨"
            if has_nature:
                return "প্রকৃতির স্নিগ্ধ রূপ অসাধারণভাবে ফুটে উঠেছে! রঙ আর সজীবতা সত্যি চমৎকার। 🌸🌿"
            if has_person:
                return "অনেক সুন্দর একটি মুহূর্ত! সবার হাসিমুখ দেখে দারুণ লাগল, শুভকামনা রইল সবসময়। 😊✨"
            return "খুবই সুন্দর এবং চমৎকার একটি ছবি! অসাধারণ ফ্রেম ও আলোছায়ার সমন্বয়। ✨"

        # ── COMPOUND & ENTITY SPECIFIC RESPONSES (ENGLISH) ──

        # 1. Person + Pet + Mountain / Trail (e.g. Woman on top of mountain with her dog)
        if (has_person and has_pet and has_mountain) or (has_pet and has_mountain):
            if tone == "enthusiastic":
                options = [
                    "Standing on top of the world with your faithful pup! Ultimate adventure goals right here! 🏔️🐾🔥",
                    "Summit conquered with the best trail buddy ever! That mountain backdrop is completely unreal! 🐶⛰️🚀",
                    "Peak paws! Conquering mountains together with your loyal companion is the greatest feeling! 🐾🏔️✨",
                ]
            elif tone == "professional":
                options = [
                    "A breathtaking high-altitude capture celebrating companionship and the spirit of outdoor exploration. Spectacular vista.",
                    "Superb alpine composition with crisp horizon depth, capturing genuine adventure and loyal companionship.",
                ]
            else:  # friendly
                options = [
                    "What an incredible adventure with your dog! Reaching the mountain summit together with that breathtaking view is pure magic. 🐾🏔️✨",
                    "Summit paws! Conquering trails and taking in the mountain breeze with your loyal companion is unbeatable. 🐶⛰️❤️",
                    "This is pure adventure bliss! That mountain vista is legendary, and your pup looks so happy by your side. 🏔️🐕✨",
                ]
            return random.choice(options)

        # 2. Person + Mountain / Hiking / Summit (no pet)
        if has_person and has_mountain:
            if tone == "enthusiastic":
                options = [
                    "Summit conquered! Nothing compares to the feeling of reaching the peak with a view like that! 🏔️🔥💪",
                    "Absolute warrior energy! That mountain backdrop looks completely epic! ⛰️🚀",
                ]
            elif tone == "professional":
                options = [
                    "An impressive alpine achievement captured with stunning landscape depth and natural light.",
                    "Exceptional high-altitude photograph showcasing personal resilience and majestic natural perspective.",
                ]
            else:
                options = [
                    "Standing on top of the world! What an incredible climb and well-earned summit view. 🏔️🙌✨",
                    "That panoramic mountain view is absolutely breathtaking! True adventure spirit at its finest. ⛰️🥾",
                ]
            return random.choice(options)

        # 3. Person + Pet (general)
        if has_person and has_pet:
            if tone == "enthusiastic":
                options = [
                    "Best duo ever! Look at that pure joy and adorable energy radiating from this picture! 🐾❤️🔥",
                    "Cutest photo on my feed today! You two make the absolute best team! 🐶✨",
                ]
            elif tone == "professional":
                options = [
                    "A heartwarming portrait highlighting genuine companionship and warmth.",
                    "A beautifully composed portrait capturing positive emotion and mutual bond.",
                ]
            else:
                options = [
                    "The bond between you two is simply heartwarming! Such a precious moment with your faithful companion. 🐶❤️",
                    "Unconditional love captured in one frame! Your furry friend looks so happy and content. 🐾✨",
                ]
            return random.choice(options)

        # 4. Pet (standalone)
        if has_pet:
            if tone == "enthusiastic":
                options = [
                    "Cutest creature on earth! Look at those eyes, I'm completely melting! 🐾🥰🔥",
                    "Absolute perfection! That adorable little face deserves endless treats and cuddles! 🐶✨",
                    "Too precious for words! Look at that adorable face and personality! 🐶❤️",
                ]
            elif tone == "professional":
                options = [
                    "A charming and well-timed pet photograph. The focal sharpness and personality shine through.",
                ]
            else:
                options = [
                    "What an adorable photo! That sweet face and innocent expression completely made my day. 🐾❤️",
                    "Such a sweet and lovely companion. Beautifully captured! 😊🐾",
                ]
            return random.choice(options)

        # 5. Mountain / Alpine (standalone)
        if has_mountain:
            if tone == "enthusiastic":
                options = [
                    "Towering grandeur! These rugged peaks and deep valleys look completely epic! ⛰️🔥",
                    "Majestic alpine peaks rising above the clouds! Absolutely phenomenal nature shot! 🏔️✨",
                ]
            elif tone == "professional":
                options = [
                    "An impressive geological landscape with excellent scale, texture, and natural contrast.",
                    "Superb alpine perspective featuring crisp ridgelines and atmospheric depth.",
                ]
            else:
                options = [
                    "The majesty of these mountain peaks is awe-inspiring! What an incredible landscape to behold. 🏔️🌲✨",
                    "Such dramatic altitude and rugged beauty! Nature at its most majestic. ⛰️✨",
                ]
            return random.choice(options)

        # 6. Sunset + Coastal / Water
        if has_sunset and (has_beach or "water" in d_low):
            if tone == "enthusiastic":
                options = [
                    "Golden hour perfection! That glowing horizon reflecting on the water looks totally surreal! 🌅✨🔥",
                    "Breathtaking twilight magic! Nature's own masterpiece reflecting across the water! 🌇🌊",
                ]
            elif tone == "professional":
                options = [
                    "A magnificent twilight study with remarkable tonal range and horizon balance.",
                ]
            else:
                options = [
                    "Golden hour reflecting over the open water is pure serenity. Absolutely gorgeous sunset colors! 🌅💙",
                    "Nature's canvas at its best! The calm water and warm twilight hues create such peaceful vibes. 🌇🌊",
                ]
            return random.choice(options)

        # 7. Sunset + Mountain
        if has_sunset and has_mountain:
            if tone == "enthusiastic":
                options = [
                    "Epic mountain sunset! That golden light hitting the ridges is pure visual perfection! 🏔️🌅🔥",
                ]
            elif tone == "professional":
                options = [
                    "Superb atmospheric landscape capture featuring crisp mountain ridgelines and warm twilight gradients.",
                ]
            else:
                options = [
                    "The sun setting behind those mountain silhouettes is breathtaking! Nature's golden masterpiece. 🏔️🌇✨",
                    "Such majestic altitude serenity! The twilight colors dancing over the peaks are unforgettable. 🌄💛",
                ]
            return random.choice(options)

        # 8. Sunset (general)
        if has_sunset:
            if tone == "enthusiastic":
                options = [
                    "Golden hour magic! That radiant sky looks like a living painting! 🌇✨🔥",
                    "Sunsets never fail to amaze! The sky looks completely on fire! 🌅🔥",
                ]
            elif tone == "professional":
                options = [
                    "Exceptional capture of atmospheric lighting and twilight color gradients.",
                ]
            else:
                options = [
                    "Sunsets never fail to amaze. Beautifully captured evening glow and tranquil skies! 🌅💛",
                ]
            return random.choice(options)

        # 9. Person + Beach / Coastal
        if has_person and has_beach:
            if tone == "enthusiastic":
                options = [
                    "Vacation paradise unlocked! That turquoise water and vibrant vibe look completely unreal! 🌊🌴🔥",
                    "Pure coastal bliss! Sun, sand, and good times—loving this energy! 🏖️☀️✨",
                ]
            elif tone == "professional":
                options = [
                    "A vibrant coastal capture reflecting balance, leisure, and idyllic natural atmosphere.",
                ]
            else:
                options = [
                    "Sun, sea breeze, and great vibes! The coastal atmosphere looks so relaxing and refreshing. 🌊☀️✨",
                    "Such radiant beach energy! There's nothing quite like unwinding by the open water. 🏖️💙",
                ]
            return random.choice(options)

        # 10. Beach / Coastal (standalone)
        if has_beach:
            if tone == "enthusiastic":
                options = [
                    "Take me there right now! That endless blue water and open sky look like total paradise! 🏖️☀️🔥",
                    "Pure vacation vibes! Loving everything about this breathtaking coastal view! 🌊✨",
                ]
            elif tone == "professional":
                options = [
                    "Superb marine perspective showing pristine horizon balance and atmospheric clarity.",
                ]
            else:
                options = [
                    "Such serene coastal vibes! There's nothing quite like the soothing beauty of the ocean. 🌊💙",
                    "What a stunning view! The tranquility and blues in this photo are amazing. ☀️",
                ]
            return random.choice(options)

        # 11. Person + Food / Dining
        if has_person and has_food:
            if tone == "enthusiastic":
                options = [
                    "Foodie heaven right here! That feast looks next-level delicious! 🤤🔥🍽️",
                    "Now THAT is a proper meal! Look at that incredible culinary spread! 😋✨",
                ]
            elif tone == "professional":
                options = [
                    "A delightful dining capture showcasing culinary appreciation and warm hospitality.",
                ]
            else:
                options = [
                    "Nothing beats sharing a great meal in wonderful company! Everything looks mouthwatering. 🍽️😋",
                    "The perfect culinary moment! Hope every bite was as delightful as it looks. ☕🍰✨",
                ]
            return random.choice(options)

        # 12. Food / Dining (standalone)
        if has_food:
            if tone == "enthusiastic":
                options = [
                    "Oh wow, that spread looks unbelievable! Making everyone hungry just looking at it! 🤤🔥✨",
                    "This looks next-level delicious! Absolutely phenomenal culinary presentation! 🔥🍽️",
                ]
            elif tone == "professional":
                options = [
                    "An appetizing culinary presentation with commendable styling and visual balance.",
                ]
            else:
                options = [
                    "This looks absolutely mouthwatering! The presentation is spot on, hope you enjoyed every bite! 🍽️😋",
                    "Such an appetizing photo! Everything looks prepared to absolute perfection. 🍽️☕",
                ]
            return random.choice(options)

        # 13. Person + Celebration / Milestone
        if has_celebration:
            if tone == "enthusiastic":
                options = [
                    "LET'S CELEBRATE! What a monumental milestone! So proud of everything you've accomplished! 🥳🎉🔥",
                    "Major congratulations! Here's to big victories and even bigger adventures ahead! 🎊✨🙌",
                ]
            elif tone == "professional":
                options = [
                    "Warmest congratulations on reaching this distinguished milestone. Wishing you continued success.",
                ]
            else:
                options = [
                    "Huge congratulations on this special moment! Celebrating milestones with loved ones creates memories to cherish forever. 🎉🥂✨",
                    "So thrilled for you! Wishing you continued happiness, success, and wonderful memories ahead. 🎊✨",
                ]
            return random.choice(options)

        # 14. Fitness / Workout / Sports
        if has_fitness:
            if tone == "enthusiastic":
                options = [
                    "Absolute beast mode! Crushing those fitness goals like a champion! 💥🏋️‍♂️🔥",
                    "Unstoppable drive! That discipline and energy are completely inspiring! 💪⚡",
                ]
            elif tone == "professional":
                options = [
                    "Exemplary focus and athletic dedication. Inspiring commitment to health and physical performance.",
                ]
            else:
                options = [
                    "Putting in the hard work and dedication! Consistency is key, and your commitment really shows. 💪🔥",
                    "Great discipline and athletic energy! Keep pushing boundaries and inspiring everyone around you. 🏃‍♂️✨",
                ]
            return random.choice(options)

        # 15. Modern Workspace / Tech
        if has_workspace:
            if tone == "enthusiastic":
                options = [
                    "Next-level workstation setup! Ready to build the future in ultimate style! 🚀💻🔥",
                ]
            elif tone == "professional":
                options = [
                    "A sophisticated, efficient workspace reflecting professionalism and creative focus.",
                ]
            else:
                options = [
                    "Dialed in and focused! Such a clean and productive setup to get big things done. 💻⚡",
                    "Great workflow and productive atmosphere! Hope the project is coming along smoothly. ☕🖥️",
                ]
            return random.choice(options)

        # 16. Vehicle / Automotive
        if has_vehicle:
            if tone == "enthusiastic":
                options = [
                    "What an absolute beast of a machine! That road presence is unreal! 🏎️🔥💨",
                    "Pure automotive eye-candy! Ready to conquer the asphalt in style! 🚗💨⚡",
                ]
            elif tone == "professional":
                options = [
                    "A crisp automotive capture with great framing and road-trip aesthetic.",
                ]
            else:
                options = [
                    "Open roads and great adventures ahead! The ride looks sharp and ready for the journey. 🚗🛣️✨",
                ]
            return random.choice(options)

        # 17. Nature / Botanical / Flower
        if has_nature:
            if tone == "enthusiastic":
                options = [
                    "Vibrant bloom perfection! Mother Nature showing off her best colors! 🌸🔥🌿",
                    "What a gorgeous capture! The details on this nature shot pop so beautifully! ✨🌺",
                ]
            elif tone == "professional":
                options = [
                    "A striking botanical capture with exceptional clarity, depth of field, and natural balance.",
                    "Excellent contrast and composition on this floral photograph. Beautiful work.",
                ]
            else:
                options = [
                    "The colors here are so vivid and lovely! What a refreshing nature photo to brighten the day. 🌸🌿",
                    "Such a peaceful botanical capture! The delicate petals and natural details really stand out. 🌺✨",
                ]
            return random.choice(options)

        # 18. People / Portrait / Group (standalone)
        if has_person:
            if tone == "enthusiastic":
                options = [
                    "Incredible energy right here! The joy and radiance in this picture are unmatched! 🎉🙌🔥",
                    "What a fantastic moment! The positive energy in this picture is completely contagious! ✨🔥",
                ]
            elif tone == "professional":
                options = [
                    "A warm and engaging photograph reflecting genuine expression and positive community spirit.",
                    "A wonderful photograph reflecting warmth, professionalism, and community.",
                ]
            else:
                options = [
                    "Love the warm and genuine vibes in this picture! Everyone looks fantastic and in great spirits. 😊✨",
                    "Such a wonderful moment captured! The happiness here is completely infectious. 💛",
                ]
            return random.choice(options)

        # 19. Dynamic archetype fallback via NLP comment generator
        try:
            from .comment_generator_model import BertCommentGeneratorModel
            generator = BertCommentGeneratorModel()
            res = generator.predict(description, style=tone, platform=platform)
            comm = res.get("comment", "").strip()
            if comm:
                return comm
        except Exception as exc:
            logger.debug("NLP comment generator fallback failed: %s", exc)

        # 20. Resilient default fallback
        if tone == "enthusiastic":
            return "Love everything about this shot! The lighting and visual energy are phenomenal! ✨📸"
        elif tone == "professional":
            return "A thoughtfully composed visual with excellent atmosphere and subject balance."
        return "Such an inviting moment with great visual appeal. Thanks for sharing this with us! ✨"

    def process_image(self, image_bytes, tone="friendly", platform="facebook", filename="", user_prompt="", language=None):
        """Run complete image understanding: describe what's happening first, then generate comment."""
        description = self.describe_image(image_bytes, filename=filename, user_prompt=user_prompt)
        comment = self.generate_comment(description, tone=tone, platform=platform, language=language)
        full_text = f"🖼️ What's happening in the image:\n{description}\n\n💬 Generated Comment:\n{comment}"

        return {
            "description": description,
            "comment": comment,
            "text": full_text,
            "model": "Salesforce/blip-image-captioning-base" if self.model else "smart-vision-analyzer",
            "source": self.source,
        }


local_vision_model = LocalVisionModel()

import io
import json
import logging
import os
import random
import re
import urllib.request
import urllib.error
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
            "pet": {"cat", "cats", "kitten", "kitty", "dog", "dogs", "puppy", "pup", "pet", "pets", "bird", "animal", "hamster", "bunny", "rabbit"},
            "food": {"food", "dish", "meal", "coffee", "cake", "pizza", "burger", "dinner", "lunch", "breakfast", "pasta", "tea", "snack", "recipe", "restaurant", "dessert"},
            "people": {"portrait", "selfie", "me", "friend", "friends", "brother", "sister", "mom", "dad", "family", "group", "team", "person", "smile", "smiling", "boy", "girl", "man", "woman"},
            "nature": {"flower", "flowers", "rose", "garden", "nature", "tree", "trees", "plant", "plants", "forest", "park", "bloom", "foliage"},
            "coastal": {"beach", "sea", "ocean", "lake", "river", "boat", "waterfront", "island", "coast", "swimming", "sand"},
            "sunset": {"sunset", "sunrise", "dusk", "dawn", "goldenhour", "evening", "twilight"},
            "celebration": {"birthday", "wedding", "anniversary", "party", "celebration", "ceremony", "graduation", "festival", "eid", "puja", "holiday"},
            "workspace": {"laptop", "desk", "office", "code", "coding", "setup", "workspace", "computer", "macbook", "study", "work"},
            "fitness": {"gym", "workout", "fitness", "run", "running", "sport", "football", "cricket", "match", "exercise"},
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

        # Build natural description — explicit semantic tags first
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

        # 2. Try Hugging Face Serverless BLIP inference if HF_TOKEN is configured
        hf_caption = self._try_hf_serverless_blip(image_bytes)
        if hf_caption:
            return hf_caption

        # 3. Advanced pure-Python visual analysis (Pillow pixel analysis + semantic tags)
        try:
            return self._pure_python_describe(img, filename=filename, user_prompt=user_prompt)
        except Exception as exc:
            logger.warning("Pure Python image description failed: %s", exc)
            w, h = img.size
            orientation = "landscape" if w > h else "portrait" if h > w else "square"
            return f"A {orientation} photograph featuring focused visual subjects in a well-balanced scene."

    def generate_comment(self, description, tone="friendly", platform="facebook"):
        """Generate a social media comment reacting specifically to what is in the image."""
        tone = (tone or "friendly").lower()
        d_low = description.lower()

        # Flower / Nature / Landscape
        if any(w in d_low for w in ["flower", "plant", "garden", "tree", "forest", "nature", "leaf", "bloom", "greenery", "foliage"]):
            if tone == "enthusiastic":
                options = [
                    "Absolutely stunning bloom! The contrast and vibrant colors here are breathtaking! 🌸🔥",
                    "What a gorgeous capture! The details on this nature shot pop so beautifully! ✨🌺",
                ]
            elif tone == "professional":
                options = [
                    "A striking botanical capture with exceptional clarity and depth. Very well framed.",
                    "Excellent contrast and composition on this floral photograph. Beautiful work.",
                ]
            else:  # friendly
                options = [
                    "The colors here are so vivid and lovely! What a refreshing photo to see. 🌸✨",
                    "Such a beautiful, peaceful capture! The vibrant natural details really stand out. 🌿",
                ]

        # Coastal / Sea / Beach
        elif any(w in d_low for w in ["beach", "sea", "ocean", "waterfront", "coastal", "lake", "water"]):
            if tone == "enthusiastic":
                options = [
                    "Take me there right now! That water view and open sky look absolutely incredible! 🌊☀️",
                    "Pure vacation paradise! Loving everything about this breathtaking coastal view! 🏖️✨",
                ]
            elif tone == "professional":
                options = [
                    "Superb coastal perspective with remarkable dynamic range and horizon balance.",
                    "A picturesque marine landscape showing excellent composition and atmospheric clarity.",
                ]
            else:
                options = [
                    "Such serene coastal vibes! There's nothing quite like the beauty of the sea. 🌊💙",
                    "What a stunning view! The tranquility and blues in this photo are amazing. ☀️",
                ]

        # People / Friends / Group / Portrait
        elif any(w in d_low for w in ["people", "person", "man", "woman", "child", "girl", "boy", "group", "crowd", "smiling", "portrait", "gathered"]):
            if tone == "enthusiastic":
                options = [
                    "Incredible energy here! Looks like an absolute blast, love seeing this! 🎉🙌",
                    "What a fantastic moment! The joy in this picture is completely contagious! ✨🔥",
                ]
            elif tone == "professional":
                options = [
                    "A great portrait capturing genuine expression and positive collaborative spirit.",
                    "A wonderful photograph reflecting warmth, professionalism, and community.",
                ]
            else:  # friendly
                options = [
                    "Love the warm vibes in this picture! Thanks so much for sharing this moment with us. 😊",
                    "Such a great photo! Everyone looks wonderful and in great spirits. ✨",
                ]

        # Animals / Pets
        elif any(w in d_low for w in ["dog", "cat", "puppy", "kitten", "bird", "animal", "pet"]):
            if tone == "enthusiastic":
                options = [
                    "Too precious for words! Look at that adorable face and personality! 🐶❤️",
                    "Cutest picture on my feed today! Absolutely loving this energy! 🐾✨",
                ]
            elif tone == "professional":
                options = [
                    "A charming and well-timed photograph. The focus and lighting are spot on.",
                ]
            else:  # friendly
                options = [
                    "What an adorable photo! Absolutely made my day to see this. 🐾❤️",
                    "Such a sweet and lovely companion. Beautifully captured! 😊",
                ]

        # Food / Coffee / Dining
        elif any(w in d_low for w in ["food", "meal", "coffee", "cake", "plate", "dish", "pizza", "table", "breakfast", "dinner", "lunch", "culinary"]):
            if tone == "enthusiastic":
                options = [
                    "Oh wow, that presentation looks unbelievable! Making me hungry just looking at it! 🤤✨",
                    "This looks next-level delicious! Absolutely phenomenal culinary spread! 🔥🍽️",
                ]
            elif tone == "professional":
                options = [
                    "An appealing culinary presentation with great attention to detail and styling.",
                ]
            else:  # friendly
                options = [
                    "This looks absolutely delicious! Hope you thoroughly enjoyed every bite. 😋☕",
                    "Such an appetizing photo! The presentation is spot on. 🍽️",
                ]

        # Sunset / Sky
        elif any(w in d_low for w in ["sunset", "sunrise", "golden-hour", "skies"]):
            if tone == "enthusiastic":
                options = [
                    "Golden hour perfection! That sky looks completely unreal! 🌅✨",
                    "Absolutely breathtaking! Nature's own masterpiece right here! 🔥",
                ]
            elif tone == "professional":
                options = [
                    "Exceptional capture of atmospheric lighting and twilight color gradients.",
                ]
            else:
                options = [
                    "Sunsets never fail to amaze. Beautifully captured evening glow! 🌅💛",
                ]

        # General / Objects / Scenes
        else:
            if tone == "enthusiastic":
                options = [
                    "Love everything about this shot! The lighting and visual energy are phenomenal! ✨📸",
                    "Such an eye-catching photo! Really stands out in the best way possible! 🔥",
                ]
            elif tone == "professional":
                options = [
                    "A thoughtfully composed visual with excellent atmosphere and subject balance.",
                    "Very clean framing and clear focal point. Appreciate you sharing this scene.",
                ]
            else:  # friendly
                options = [
                    "Such an inviting moment with great visual appeal. Thanks for sharing this with us! ✨",
                    "Really love the perspective and mood captured here. Beautiful photo! 😊",
                ]

        return random.choice(options)

    def process_image(self, image_bytes, tone="friendly", platform="facebook", filename="", user_prompt=""):
        """Run complete image understanding: describe what's happening first, then generate comment."""
        description = self.describe_image(image_bytes, filename=filename, user_prompt=user_prompt)
        comment = self.generate_comment(description, tone=tone, platform=platform)
        full_text = f"🖼️ What's happening in the image:\n{description}\n\n💬 Generated Comment:\n{comment}"

        return {
            "description": description,
            "comment": comment,
            "text": full_text,
            "model": "Salesforce/blip-image-captioning-base" if self.model else "smart-vision-analyzer",
            "source": self.source,
        }


local_vision_model = LocalVisionModel()

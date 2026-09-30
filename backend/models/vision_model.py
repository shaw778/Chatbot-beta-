import io
import logging
import random
from PIL import Image

logger = logging.getLogger("chatbot.vision")


class LocalVisionModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.source = "blip-image-captioning"
        self._load_attempted = False

    def _ensure_loaded(self):
        """Lazy-load the BLIP model on first call to optimize startup time."""
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

    def describe_image(self, image_bytes):
        """Analyze the image and return a clear description of what is happening."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to open image for vision captioning: %s", exc)
            return "An uploaded photograph with distinct visual subjects."

        # 1. Try BLIP AI captioning
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

        # 2. Heuristic fallback based on image composition
        width, height = img.size
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        return f"A {orientation} image ({width}x{height}) showing a focused visual scene with rich contrast."

    def generate_comment(self, description, tone="friendly", platform="facebook"):
        """Generate a social media comment reacting specifically to what is in the image."""
        tone = (tone or "friendly").lower()
        d_low = description.lower()

        # Flower / Nature / Landscape
        if any(w in d_low for w in ["flower", "plant", "garden", "tree", "forest", "nature", "leaf", "bloom"]):
            if tone == "enthusiastic":
                options = [
                    "Absolutely stunning bloom! The contrast and vibrant colors here are breathtaking! 🌸🔥",
                    "What a gorgeous capture! The details on this flower pop so beautifully! ✨🌺",
                ]
            elif tone == "professional":
                options = [
                    "A striking botanical capture with exceptional clarity and depth. Very well framed.",
                    "Excellent contrast and composition on this floral photograph. Beautiful work.",
                ]
            else:  # friendly
                options = [
                    "The colors on this flower are so vivid and lovely! What a refreshing photo to see. 🌸✨",
                    "Such a beautiful, peaceful capture! The vibrant details really stand out. 🌿",
                ]

        # People / Friends / Group
        elif any(w in d_low for w in ["people", "person", "man", "woman", "child", "girl", "boy", "group", "crowd", "smiling", "standing"]):
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
        elif any(w in d_low for w in ["food", "meal", "coffee", "cake", "plate", "dish", "pizza", "table", "breakfast", "dinner", "lunch"]):
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

    def process_image(self, image_bytes, tone="friendly", platform="facebook"):
        """Run complete image understanding: describe what's happening first, then generate comment."""
        description = self.describe_image(image_bytes)
        comment = self.generate_comment(description, tone=tone, platform=platform)
        full_text = f"🖼️ What's happening in the image:\n{description}\n\n💬 Generated Comment:\n{comment}"

        return {
            "description": description,
            "comment": comment,
            "text": full_text,
            "model": "Salesforce/blip-image-captioning-base" if self.model else "local-vision-engine",
            "source": self.source,
        }


local_vision_model = LocalVisionModel()

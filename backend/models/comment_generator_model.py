import random
import re
import time
from pathlib import Path

from ..config import COMMENT_GENERATOR_MODEL_PATH
from .text_features import detect_language, extract_dynamic_topics, hashtags, is_question_intent, tokens


class BertCommentGeneratorModel:
    def __init__(self):
        self.model_path = Path(COMMENT_GENERATOR_MODEL_PATH)
        self.pipeline = None
        self.source = "nlp-generative-engine"
        self._load()

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text2text-generation",
                model=str(self.model_path),
                tokenizer=str(self.model_path),
            )
            self.source = "transformers-bert"
        except Exception:
            self.pipeline = None
            self.source = "nlp-generative-engine"

    def predict(self, text, sentiment="neutral", toxicity=False, style="casual", platform="general", archetype_idx=0, language=None):
        """Generate a single high-quality comment adapted to the post context, sentiment, and style."""
        if not language:
            language = detect_language(text)

        # 1. Check if input is a nested comment-reply prompt
        reply_comment = self._check_comment_reply_prompt(text, language=language, sentiment=sentiment)
        if reply_comment:
            return {
                "model": "bert-comment-generator" if self.pipeline else "nlp-comment-generator",
                "comment": reply_comment,
                "language": language,
                "sentiment": sentiment,
                "toxic": toxicity,
                "source": self.source,
                "explanation": "Contextual reply generated for the user's Facebook comment.",
            }

        # 2. Local transformer model if available
        if self.pipeline:
            try:
                prompt = (
                    f"Write one natural, human-sounding social media reply in {language} for this post: {text[:300]} "
                    "React to one specific detail, use conversational wording, and avoid generic praise, AI cliches, "
                    "hashtags, and claims of personal experience. Keep it to one or two sentences."
                )
                generated = self.pipeline(prompt, max_length=64, do_sample=True)[0].get("generated_text", "")
                if generated and len(generated.strip()) > 8:
                    return {
                        "model": "bert-comment-generator",
                        "comment": generated.strip(),
                        "language": language,
                        "sentiment": sentiment,
                        "toxic": toxicity,
                        "source": "transformers-bert",
                        "explanation": f"Generated with a local transformer model for {language}.",
                    }
            except Exception:
                pass

        # 3. Dynamic Generative Engine
        comment = self._generate_archetype_comment(text, language, sentiment, toxicity, style, platform, archetype_idx)
        return {
            "model": "nlp-comment-generator",
            "comment": comment,
            "language": language,
            "sentiment": sentiment,
            "toxic": toxicity,
            "source": self.source,
            "explanation": f"Generated contextual social comment for {language}.",
        }

    def generate_multiple(self, text, sentiment="neutral", toxicity=False, style="casual", platform="general", num_comments=3, language=None):
        """Generate N diverse, distinct comments across unique perspectives and archetypes."""
        if not language:
            language = detect_language(text)
        n = max(1, min(10, int(num_comments or 3)))
        comments = []
        used_texts = set()

        for i in range(n):
            pred = self.predict(
                text,
                sentiment=sentiment,
                toxicity=toxicity,
                style=style,
                platform=platform,
                archetype_idx=i,
                language=language,
            )
            comm = pred.get("comment", "")
            if comm and comm not in used_texts:
                used_texts.add(comm)
                comments.append(comm)

        # Ensure we meet the count with guaranteed uniqueness
        while len(comments) < n:
            extra = self._generate_archetype_comment(text, language, sentiment, toxicity, style, platform, len(comments) + random.randint(1, 99))
            if extra not in used_texts:
                used_texts.add(extra)
                comments.append(extra)
            else:
                break

        return {
            "model": "nlp-comment-generator",
            "comments": comments,
            "language": language,
            "source": self.source,
        }

    def _check_comment_reply_prompt(self, text, language=None, sentiment="neutral"):
        """Detect and handle comment reply instructions from the UI for English and Bangla."""
        if "Someone commented on a Facebook post" in text:
            match = re.search(r'Comment:\s*"(.*?)"', text, re.DOTALL)
            comment_text = match.group(1).strip() if match else ""
            c_low = comment_text.lower()

            # Check if comment or requested language is Bangla
            is_bangla = (language == "bn") or (detect_language(comment_text) == "bn") or any("\u0980" <= c <= "\u09FF" for c in comment_text)

            if is_bangla:
                # 0. Negative sentiment / complaints / issues
                is_neg_bn = (str(sentiment).lower() == "negative") or any(w in c_low for w in ["খারাপ", "বাজে", "সমস্যা", "ধোঁকা", "প্রতারণা", "নষ্ট", "দেরি", "হতাশ", "ক্ষতি", "ভুল", "অভিযোগ", "কাজ করে না", "ফালতু", "অসন্তুষ্ট"])
                if is_neg_bn:
                    replies = [
                        "আপনার এই অনাকাঙ্ক্ষিত ও হতাশাজনক অভিজ্ঞতার জন্য আমরা আন্তরিকভাবে দুঃখিত। বিষয়টি দ্রুত খতিয়ে দেখে সমাধানের জন্য অনুগ্রহ করে বিস্তারিত তথ্য দিয়ে আমাদের পেজে ইনবক্স করুন।",
                        "আমরা আপনার ভোগান্তির জন্য আন্তরিকভাবে ক্ষমাপ্রার্থী। আমরা আপনার ক্ষোভ বুঝতে পারছি এবং দ্রুত সমাধান দিতে প্রস্তুত। দয়া করে ইনবক্সে আমাদের সাথে যোগাযোগ করুন।",
                        "যে অসুবিধার সম্মুখীন হয়েছেন তার জন্য আমরা গভীরভাবে দুঃখ প্রকাশ করছি। এই সমস্যা অবিলম্বে সমাধানে আমাদের টিম প্রস্তুত, অনুগ্রহ করে বিস্তারিত ইনবক্সে দিন।",
                    ]
                    return random.choice(replies)
                # 1. Price / Order / Product inquiry
                if any(w in c_low for w in ["দাম", "কত", "কতো", "টাকা", "প্রাইস", "কবে", "পাবো", "পাব", "অর্ডার", "কিভাবে", "কীভাবে", "ডিটেইলস", "ইনবক্স", "price", "cost", "dam", "koto", "inbox"]):
                    replies = [
                        "ধন্যবাদ আপনার আগ্রহের জন্য! বিস্তারিত জানতে অনুগ্রহ করে আমাদের পেজে ইনবক্স করুন, আমরা দ্রুত উত্তর দেব।",
                        "পণ্যটির মূল্য ও অর্ডারের নিয়ম ইনবক্সে পাঠিয়ে দেওয়া হয়েছে। অনুগ্রহ করে ইনবক্স চেক করুন!",
                        "ধন্যবাদ! পণ্যটি অর্ডার করতে বা বিস্তারিত জানতে আমাদের পেজে একটি মেসেজ দিন অথবা ইনবক্সে নক দিন।",
                    ]
                # 2. Congratulations & compliments
                elif any(w in c_low for w in ["অভিনন্দন", "শুভকামনা", "দারুণ", "দারুন", "সুন্দর", "সেরা", "মাশাল্লাহ", "চমৎকার", "অসাধারণ", "অনেক সুন্দর", "ভালো", "ধন্যবাদ", "congrat", "congrats", "shundor", "valo", "bhalo"]):
                    replies = [
                        "অনেক অনেক ধন্যবাদ আপনার সুন্দর মন্তব্যের জন্য! আপনার ভালোবাসা আমাদের অনুপ্রেরণা। ❤️",
                        "অসংখ্য ধন্যবাদ! আপনাদের এমন সমর্থন ও প্রশংসায় আমরা অত্যন্ত আনন্দিত। শুভকামনা রইল!",
                        "আন্তরিক ধন্যবাদ আপনার ইতিবাচক মতামতের জন্য! সবসময় পাশে থাকবেন আশা করি। ✨",
                    ]
                # 3. Agreement / confirmation
                elif any(w in c_low for w in ["একমত", "সত্যি", "সঠিক", "ঠিক", "একদম"]):
                    replies = [
                        "একদম ঠিক বলেছেন! আপনার মূল্যবান মতামতের জন্য অসংখ্য ধন্যবাদ।",
                        "আপনার সাথে পুরোপুরি একমত। সুন্দরভাবে তুলে ধরার জন্য ধন্যবাদ!",
                        "ঠিক তাই! আমাদের পোস্টে আপনার গঠনমূলক মতামত দেওয়ার জন্য কৃতজ্ঞতা।",
                    ]
                # 4. General Bangla comment
                else:
                    replies = [
                        "আমাদের পোস্টে মূল্যবান মন্তব্য করার জন্য অসংখ্য ধন্যবাদ! সাথে থাকুন।",
                        "আপনার সুন্দর মতামতের জন্য আন্তরিক ধন্যবাদ! দিনটি শুভ হোক।",
                        "ধন্যবাদ সাথে থাকার জন্য! যেকোনো তথ্য বা সহযোগিতার জন্য আমরা আছি পাশে।",
                    ]
                return random.choice(replies)

            # English replies
            # 0. Negative sentiment / complaints / issues
            is_neg_en = (str(sentiment).lower() == "negative") or any(w in c_low for w in [
                "bad", "worst", "terrible", "horrible", "broken", "issue", "problem", "disappointed",
                "disappointing", "poor", "scam", "waste", "failed", "crash", "bug", "hate", "sucks", "cheat", "fake", "delayed"
            ])
            if is_neg_en:
                replies = [
                    "We sincerely apologize for this frustrating experience. This is certainly not our standard, and we want to make things right. Please send us a direct message so we can resolve this immediately.",
                    "We are truly sorry to hear about your experience. We understand your frustration and would like to look into this right away. Please reach out to our support team directly.",
                    "Thank you for bringing this to our attention. We apologize for the inconvenience and frustration caused, and we are committed to resolving this issue promptly. Please send us a direct message.",
                ]
                return random.choice(replies)

            if any(w in c_low for w in ["congrat", "congrats", "well done", "awesome", "great"]):
                replies = [
                    "Thank you so much! Really appreciate the kind words and support.",
                    "Thanks for celebrating with us! Means a lot.",
                    "Appreciate you taking the time to share some love!",
                ]
            elif "?" in comment_text or any(w in c_low for w in ["what", "how", "when", "why"]):
                replies = [
                    "Great question! We will be sharing more details very soon, stay tuned.",
                    "Appreciate you asking — feel free to drop us a direct message and we can share more context!",
                    "Thanks for checking in! More exciting updates on this are coming up.",
                ]
            elif any(w in c_low for w in ["agree", "true", "facts", "spot on", "right"]):
                replies = [
                    "Spot on! Couldn't have said it better myself.",
                    "Totally agree with you on that point. Thanks for chiming in!",
                    "100% with you here. Appreciate your perspective!",
                ]
            else:
                replies = [
                    "Thanks for joining the conversation and sharing your thoughts!",
                    "Appreciate the feedback and engagement on this!",
                    "Thanks for the comment! Great to have you as part of the community.",
                ]
            return random.choice(replies)
        return None

    def _extract_subject_and_detail(self, text):
        """Extract focal entity, key action, and specific detail from the text."""
        lower = text.lower()
        topics = extract_dynamic_topics(text)
        main_topic = topics[0] if topics else "this update"

        # Multi-Category Intent Classification
        is_question = is_question_intent(text)
        is_food = any(w in lower for w in [
            "recipe", "cooked", "cooking", "baked", "baking", "food", "dinner", "lunch", "breakfast",
            "delicious", "biryani", "pizza", "burger", "pasta", "ramen", "coffee", "dessert", "cake",
            "meal", "snack", "dish", "homemade", "taste", "tasty", "রান্না", "খাবার", "বিরিয়ানি", "চা", "কফি", "রেসিপি", "নাস্তা"
        ])
        is_pet = any(w in lower for w in [
            "dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "kittens", "pet", "pets",
            "rescue", "adopted", "paws", "furry", "barking", "meow", "fluffy", "shedding", "কুকুর", "বিড়াল", "বিড়ালছানা", "বিড়াল", "পোষা প্রাণী"
        ])
        is_opinion = any(w in lower for w in [
            "unpopular opinion", "hot take", "i think", "i believe", "in my opinion", "honestly",
            "debate", "agree or disagree", "prefer", "better than", "is overrated", "is underrated",
            "আমার মতে", "মনে হয়", "বিতর্ক", "ব্যক্তিগতভাবে"
        ])
        is_sports = any(w in lower for w in [
            "victory", "win", "final over", "ilt20", "championship", "cup", "trophy", "match", "game",
            "tournament", "vipers", "scored", "wickets", "runs", "goal", "জয়", "বিজয়", "ম্যাচ", "খেলা",
            "টুর্নামেন্ট", "ট্রফি", "উইকেট", "রান", "গোল"
        ])
        is_milestone = any(w in lower for w in [
            "marathon", "graduated", "promoted", "degree", "anniversary", "birthday", "celebrating", "years",
            "achievement", "journey", "completed", "training", "passed", "milestone", "মাইলফলক", "গ্র্যাজুয়েশন", "প্রমোশন", "জন্মদিন",
            "বার্ষিকী", "সাফল্য", "অর্জিত", "অর্জন", "পরিশ্রম"
        ])
        is_travel_or_delay = any(w in lower for w in [
            "flight", "luggage", "airport", "delayed", "delay", "commute", "train", "traffic", "stuck in",
            "ফ্লাইট", "দেরি", "ট্রেন", "যানজট", "জ্যাম"
        ])
        is_struggle = is_travel_or_delay or bool(
            re.search(
                r"\b(?:sadly|tough|loss|struggling|frustrated|broken|missed|bad|worst|nightmare|ruined|exhausted|painful|horrible)\b",
                lower,
            )
        ) or any(w in lower for w in ["খারাপ", "কষ্ট", "হতাশ", "সমস্যা", "নষ্ট"])
        is_tech = any(w in lower for w in [
            "launch", "launched", "product", "release", "app", "feature", "code", "ai", "platform", "version",
            "update", "github", "software", "dashboard", "developer", "programming", "python", "javascript",
            "frontend", "backend", "deploy", "database", "api", "bug", "debugging", "css", "লঞ্চ", "প্রজেক্ট", "পণ্য", "প্রোডাক্ট", "অ্যাপ", "ওয়েবসাইট",
            "কোড", "এআই", "আপডেট", "ভার্সন", "উদ্বোধন", "সফটওয়্যার"
        ])
        is_creative = any(w in lower for w in [
            "photo", "picture", "art", "sunset", "travel", "shot", "view", "aesthetic", "music", "design",
            "scenery", "landscape", "sketch", "painting", "ছবি", "আর্ট", "শিল্প", "দৃশ্য", "ভিডিও", "গান", "ডিজাইন", "ফটোগ্রাফি"
        ])

        # Priority order
        if is_struggle:
            category = "struggle"
        elif is_question:
            category = "question_advice"
        elif is_food:
            category = "food"
        elif is_pet:
            category = "pet"
        elif is_opinion:
            category = "opinion"
        elif is_sports:
            category = "sports"
        elif is_milestone:
            category = "milestone"
        elif is_tech:
            category = "tech"
        elif is_creative:
            category = "creative"
        else:
            category = "casual"

        # Context-aware natural subject phrasing
        is_bangla_text = detect_language(text) == "bn"
        if is_bangla_text:
            if category == "question_advice":
                subject = "এই বিষয়ে সিদ্ধান্ত"
            elif category == "food":
                subject = "রান্না করা খাবার" if "রান্না" in lower else "এই সুস্বাদু খাবার"
            elif category == "pet":
                subject = "এই আদুরে বিড়ালছানা" if "বিড়াল" in lower or "বিড়াল" in lower else "এই আদুরে পোষা প্রাণী"
            elif category == "opinion":
                subject = "আপনার এই মতামত"
            elif category == "sports":
                subject = topics[0] if (topics and topics[0] not in {"general", "social update"}) else "ম্যাচ জয়"
            elif category == "tech":
                subject = topics[0] if (topics and topics[0] not in {"general", "social update"}) else "এই চমৎকার প্রজেক্ট"
            elif category == "milestone":
                subject = topics[0] if (topics and topics[0] not in {"general", "social update"}) else "এই স্মরণীয় মাইলফলক"
            elif category == "struggle":
                subject = "এই অপ্রত্যাশিত পরিস্থিতি"
            elif category == "creative":
                subject = "এই চমৎকার দৃশ্য"
            else:
                subject = "এই সুন্দর মুহূর্ত"
        elif is_sports and any(v in lower for v in ["vipers", "desert vipers"]):
            subject = "Desert Vipers"
        elif category == "question_advice":
            if "python" in lower and "javascript" in lower:
                subject = "choosing between Python and JavaScript"
            elif "interview" in lower:
                subject = "interview preparation"
            elif "remote" in lower or "wfh" in lower:
                subject = "remote work productivity"
            elif "start" in lower or "learn" in lower:
                subject = "getting started on this path"
            elif topics and topics[0] != "social update":
                subject = topics[0]
            else:
                subject = "this question"
        elif category == "food":
            if "biryani" in lower:
                subject = "this homemade biryani"
            elif "ramen" in lower:
                subject = "this delicious ramen"
            elif "pizza" in lower:
                subject = "this pizza"
            elif "baking" in lower or "bread" in lower or "cookies" in lower:
                subject = "freshly baked treats"
            elif topics and topics[0] != "social update":
                subject = f"{topics[0]} dish"
            else:
                subject = "this delicious meal"
        elif category == "pet":
            if "puppy" in lower or "dog" in lower:
                subject = "your adorable pup"
            elif "kitten" in lower or "cat" in lower:
                subject = "your cute cat"
            else:
                subject = "your furry friend"
        elif category == "opinion":
            if "remote" in lower:
                subject = "remote vs in-office work"
            elif topics and topics[0] != "social update":
                subject = f"your perspective on {topics[0]}"
            else:
                subject = "your perspective"
        elif is_milestone and "marathon" in lower:
            subject = "completing the marathon"
        elif is_milestone and "graduated" in lower:
            subject = "graduating"
        elif is_milestone and "promoted" in lower:
            subject = "the promotion"
        elif is_tech and "dashboard" in lower:
            subject = "the new AI dashboard"
        elif is_tech and "app" in lower:
            subject = "the new app"
        elif is_travel_or_delay and "flight" in lower:
            subject = "the flight delay"
        elif is_travel_or_delay and "luggage" in lower:
            subject = "the lost luggage"
        elif is_travel_or_delay and "traffic" in lower:
            subject = "the brutal traffic delay"
        elif topics and len(topics[0].split()) > 1:
            subject = topics[0]
        elif len(topics) >= 2 and len(topics[0]) <= 4:
            subject = f"{topics[0]} {topics[1]}"
        elif topics and topics[0] != "social update":
            subject = topics[0]
        else:
            subject = "this issue" if (category == "struggle" or is_struggle) else "this moment"

        # Extract concrete detail
        specific_detail = None
        detail_patterns = [
            r"\b(?:after|during|in)\s+an?\s+([a-zA-Z0-9\s]+?)(?:!|\.|\,|$)",
            r"\b(?:victory|win|triumph)\s+in\s+([a-zA-Z0-9\s]+?)(?:!|\.|\,|$)",
            r"\b(?:launch|release)\s+of\s+([a-zA-Z0-9\s]+?)(?:!|\.|\,|$)",
            r"\b(?:claim(?:ed|s)?|won|secured)\s+([a-zA-Z0-9\s]+?)(?:!|\.|\,|$)",
        ]
        for pat in detail_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                extracted = m.group(1).strip()
                if 3 < len(extracted) < 40 and not any(extracted.lower().startswith(s) for s in ["finally", "flight", "just"]):
                    specific_detail = extracted
                    break

        if not specific_detail:
            specific_detail = subject

        return subject, specific_detail, category

    def _generate_archetype_comment(self, text, language, sentiment, toxicity, style, platform, archetype_idx):
        """Generate comment based on 6 distinct conversational archetypes adapted to context and category."""
        if toxicity:
            return "I appreciate you bringing this topic forward, and hope we can keep the discussion respectful and constructive for everyone."

        subject, detail, category = self._extract_subject_and_detail(text)
        archetypes = ["celebratory", "discussion_question", "detail_reactor", "punchy_social", "observant_pro", "warm_community"]
        selected_archetype = archetypes[archetype_idx % len(archetypes)]

        # Multilingual handling for non-English posts
        if language != "en" and language in ["bn", "es", "fr", "de", "pt", "it", "hi", "ar", "zh", "ja"]:
            return self._multilingual_comment(language, subject, category, sentiment, archetype_idx)

        # Handle negative sentiment posts (e.g. empathy, bounce back, support)
        if sentiment == "negative" or (category == "struggle" and sentiment != "positive"):
            return self._generate_negative_support(subject, detail, selected_archetype)

        # ── 1. QUESTION & ADVICE INQUIRIES ─────────────────────────
        if category == "question_advice":
            if selected_archetype in ["celebratory", "detail_reactor"]:
                templates = [
                    f"Both paths have solid merits, but starting with the core fundamentals and building small hands-on projects will give you the fastest confidence boost!",
                    f"Great question to ask upfront! My recommendation is to start with the option that has the most active community and relevant tutorials for your exact goals.",
                    f"You can't go wrong taking it step by step. Building a few practical mini-projects around {subject} will clarify your direction faster than anything else.",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"Really thoughtful question! What specific project or use-case are you most excited to tackle once you dive into {subject}?",
                    f"Curious to hear what led to this inquiry — are you looking to solve a specific problem or exploring new career horizons?",
                    f"Such an important decision! Have you had a chance to experiment with small samples of either yet?",
                ]
            elif selected_archetype == "punchy_social":
                templates = [
                    f"Go for it! The learning curve is part of the fun, and you'll pick up momentum super fast. 🚀💯",
                    f"Take the leap! You're going to learn a ton along the way. Excited for your journey! ✨",
                    f"Jump right in! Hands-on building is always the best teacher. 🙌",
                ]
            elif selected_archetype == "observant_pro":
                templates = [
                    f"From practical experience, the core problem-solving principles around {subject} will serve you well no matter which specific tools you settle on.",
                    f"A structured, project-driven approach to {subject} will always yield the best return on your time. Focus on building rather than just consuming theory.",
                ]
            else:  # warm_community
                templates = [
                    f"Rooting for you as you explore {subject}! Don't hesitate to reach out if you hit roadblocks along the way.",
                    f"Such an exciting step forward. The community is right behind you — enjoy every bit of the learning curve! 🙌",
                ]

        # ── 2. FOOD & DINING ───────────────────────────────────────
        elif category == "food":
            if selected_archetype in ["celebratory", "punchy_social"]:
                templates = [
                    f"Now that looks absolutely mouthwatering! Nothing beats the satisfaction of {subject} done to perfection. 🍽️🤤",
                    f"Look at that presentation! You can practically smell how delicious {subject} is right through the screen. 😋",
                    f"Chef-level work right here! {subject.title()} looks unbelievably tasty — definitely making me hungry. ✨",
                    f"10/10 plate right here! Send a portion over please! 🤤🍽️",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"This looks incredible! What’s your secret spice or special twist that makes your {subject} turn out so well?",
                    f"Pure comfort food! Did you follow a classic recipe for {subject} or experiment with your own variation?",
                ]
            elif selected_archetype == "detail_reactor":
                templates = [
                    f"The color and texture on {subject} look spot on. It’s clear a lot of care went into making this.",
                    f"The balance of flavors and presentation on {subject} really stand out here.",
                ]
            else:  # observant_pro / warm_community
                templates = [
                    f"Meals like {subject} bring people together in the warmest way. Hope everyone at the table savored every single bite! ❤️",
                    f"Cooking with this level of attention is a true art. Thanks for sharing this delicious moment!",
                ]

        # ── 3. PETS & ANIMALS ──────────────────────────────────────
        elif category == "pet":
            if selected_archetype in ["celebratory", "warm_community"]:
                templates = [
                    f"Look at that precious face! You can never stay mad at them for long, no matter what mischief they get into. 🐾❤️",
                    f"Total cuteness overload! Pets have an unmatched superpower for making any day instantly brighter. 🐶✨",
                    f"Those eyes say it all! Thanks for sharing this adorable moment with {subject} — brought an instant smile to my feed. ✨",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"Too adorable! How old is this little companion, and what’s their favorite habit or toy so far? 🐾",
                    f"Such a sweet little soul! How did {subject} come into your life?",
                ]
            elif selected_archetype == "punchy_social":
                templates = [
                    f"Protect this little angel at all costs! 🐾🥺✨",
                    f"The absolute cutest! Can't handle this level of adorable. 🥰",
                    f"Instant mood booster right here! Absolutely precious. ❤️",
                ]
            else:  # detail_reactor / observant_pro
                templates = [
                    f"Furry companions bring so much pure joy into our homes. Wishing you both countless happy memories together! 🐾❤️",
                    f"That joyful expression on {subject} says everything. Unconditional love at its finest!",
                ]

        # ── 4. OPINION & DISCUSSION ────────────────────────────────
        elif category == "opinion":
            if selected_archetype in ["observant_pro", "celebratory"]:
                templates = [
                    f"You raise a really sharp perspective on {subject}. There are strong arguments on both sides, but this highlights a nuance people often miss.",
                    f"Appreciate this grounded take on {subject}. It really comes down to trade-offs and what aligns best with individual priorities.",
                    f"Spot on analysis! Having thoughtful, balanced discussions around {subject} is exactly what drives the community forward.",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"Compelling argument! What was the defining experience or observation that solidified your stance on {subject}?",
                    f"Such an interesting angle on {subject}. How do you see this evolving over the next few years?",
                ]
            elif selected_archetype == "punchy_social":
                templates = [
                    f"Underrated take! Really glad someone said this out loud. 💯",
                    f"Hard agree on this! You hit the nail on the head. 👏",
                ]
            else:  # detail_reactor / warm_community
                templates = [
                    f"Thanks for sharing such an engaging perspective on {subject}. Love seeing insightful discussions like this on the feed!",
                    f"The points you outlined around {subject} are well reasoned and hard to dispute. Appreciate you posting this!",
                ]

        # ── 5. SPORTS ──────────────────────────────────────────────
        elif category == "sports":
            if selected_archetype in ["celebratory", "punchy_social"]:
                templates = [
                    f"Massive congratulations to {subject}! What an extraordinary performance right down to the wire. 🏆👏",
                    f"Incredible finish! {subject} truly stepped up when the pressure was on. Fully deserved triumph! 🔥",
                    f"What a phenomenal victory for {subject}! That kind of clutch performance is what champions are made of.",
                    f"Pure class from {subject}! That final stretch was absolute madness. 🔥🙌",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"That finish had everyone on the edge of their seats! What do you think was the defining moment for {subject}?",
                    f"Incredible game from start to finish. How do you rate this performance compared to earlier matches this season?",
                ]
            elif selected_archetype == "detail_reactor":
                templates = [
                    f"The composure shown during {detail} was top-tier. Really sets this apart from anything else we’ve seen.",
                    f"That decisive move during {detail} proved to be the game-changer. Sensational execution!",
                ]
            else:
                templates = [
                    f"A thoroughly well-earned result for {subject}. The strategy and follow-through were textbook. 🏆",
                    f"Moments like this are exactly why we follow {subject} so passionately. Thrilled for the team! ✨",
                ]

        # ── 6. TECH & DEVELOPMENT ──────────────────────────────────
        elif category == "tech":
            if selected_archetype in ["celebratory", "punchy_social"]:
                templates = [
                    f"Huge congratulations on {subject}! Awesome to see this milestone come to life. 🚀",
                    f"Big win for the whole team behind {subject}! The hard work and engineering craftsmanship really shine through.",
                    f"So exciting to see {subject} rolling out! Wishing you massive momentum with this launch. ✨",
                    f"Big moves with {subject}! Clean execution. 🚀💯",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"Really interesting approach with {subject}. What was the most challenging technical hurdle during development?",
                    f"This looks super promising! What’s the next big feature or update on the roadmap for {subject}?",
                ]
            elif selected_archetype == "detail_reactor":
                templates = [
                    f"The execution on {subject} was spot on. That level of focus is rare and hard to miss.",
                    f"The architecture and clean design behind {subject} really stand out here. Super impressive work.",
                ]
            else:
                templates = [
                    f"A fantastic case study in discipline and product craft. {subject.title()} sets a very high benchmark.",
                    f"Always exciting seeing dedication and engineering excellence pay off like this for {subject}!",
                ]

        # ── 7. MILESTONE CELEBRATIONS ──────────────────────────────
        elif category == "milestone":
            if selected_archetype in ["celebratory", "punchy_social"]:
                templates = [
                    f"Huge congratulations on {subject}! Such an inspiring milestone after all that dedication. 👏🏅",
                    f"Finishing strong! {subject.title()} is incredible proof of consistency and dedication. Fully deserved! 🔥",
                    f"What an awesome achievement with {subject}! Celebrating right along with you. 🎉",
                    f"Legendary effort on {subject}! Huge respect. 🏅🔥",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"That takes serious dedication! What was the toughest part of the journey leading up to {subject}?",
                    f"Such an inspiring milestone! What was the biggest lesson you learned throughout {subject}?",
                ]
            else:
                templates = [
                    f"A thoroughly well-earned result for {subject}. Proves that persistent effort always pays dividends.",
                    f"Moments like this are what make the journey so rewarding. Wishing you even greater success ahead! ✨",
                ]

        # ── 8. CREATIVE & PHOTOGRAPHY ──────────────────────────────
        elif category == "creative":
            if selected_archetype in ["celebratory", "punchy_social"]:
                templates = [
                    f"Stunning aesthetic and composition! The light and atmosphere captured in {subject} are truly mesmerizing. 📸🎨",
                    f"This has such an evocative, captivating vibe. Absolutely wonderful work on {subject}!",
                    f"Pure visual poetry! Loving the tones and perspective here. ✨",
                ]
            elif selected_archetype == "discussion_question":
                templates = [
                    f"Beautiful eye for detail! Which lens or setup did you use to capture this shot of {subject}?",
                    f"Incredible mood here! What was your inspiration when creating {subject}?",
                ]
            else:
                templates = [
                    f"The lighting and color grading on {subject} are masterclass. Really impressive artistic vision.",
                    f"Such an inspiring piece. Thanks for sharing this creative moment with the community! 🌸",
                ]

        # ── 9. CASUAL & LIFESTYLE ──────────────────────────────────
        else:
            templates = [
                f"Such a peaceful, uplifting update. Hope you get to relax and enjoy every bit of this moment! ☕✨",
                f"Love the calm energy here. It’s the simple moments like {subject} that make the week so worthwhile.",
                f"Wishing you a wonderful, refreshing time ahead! Thanks for sharing this slice of positivity.",
                f"This brought a genuine smile to my feed today. Thanks so much for sharing {subject}!",
                f"Appreciate this warm update! Hope the rest of your week is just as pleasant.",
            ]

        # Select template using stable modulo + variation
        seed_offset = (archetype_idx * 7 + len(text)) % len(templates)
        comment = templates[seed_offset]
        return self._format_for_platform(comment, platform)

    def _generate_negative_support(self, subject, detail, archetype):
        """Generate professional, apologetic, and empathetic customer-service crisis management replies for negative situations."""
        clean_subj = subject.strip() if subject else "this matter"
        if archetype in ["celebratory", "punchy_social"]:
            templates = [
                f"We sincerely apologize for your frustrating experience with {clean_subj}. This is certainly not the standard we aim for. Please send us a direct message so our team can investigate and resolve this for you immediately.",
                f"I am truly sorry to hear about the trouble you've experienced with {clean_subj}. We acknowledge your frustration and want to make this right. Please reach out to our support team directly.",
                f"We deeply apologize for the inconvenience and frustration caused by {clean_subj}. Your satisfaction is very important to us, and we are committed to resolving this issue promptly. Please contact us directly.",
            ]
        elif archetype == "discussion_question":
            templates = [
                f"We are very sorry for the frustration caused by {clean_subj}. Could you please send us a direct message with your details so we can investigate and resolve this for you right away?",
                f"We sincerely apologize for this disappointing experience with {clean_subj}. Please let us know how our team can best assist you in resolving this matter immediately.",
            ]
        else:
            templates = [
                f"We sincerely apologize for the frustration regarding {clean_subj}. We understand how difficult this is, and our team is ready to step in and resolve the issue for you immediately.",
                f"I am so sorry for your negative experience with {clean_subj}. We take your feedback seriously and would appreciate the opportunity to make things right. Please send us a direct message.",
                f"We apologize for falling short of your expectations with {clean_subj}. We acknowledge your frustration and want to assist in resolving this. Please connect with our support team so we can help.",
            ]
        return random.choice(templates)

    def _format_for_platform(self, comment, platform):
        """Fine-tune comment length and formatting for specific social platforms."""
        p = str(platform or "general").lower()
        if p in ["x", "twitter"]:
            # Keep under 200 chars for concise tweet replies
            if len(comment) > 180:
                sentences = comment.split(". ")
                return sentences[0] + "." if sentences else comment[:180]
        return comment

    def _multilingual_comment(self, lang, subject, category, sentiment, archetype_idx):
        """Generate culturally natural replies for non-English posts."""
        if lang == "bn":
            if sentiment == "negative" or category == "struggle":
                bn_neg = [
                    f"{subject} নিয়ে আপনার এই অনাকাঙ্ক্ষিত ও হতাশাজনক অভিজ্ঞতার জন্য আমরা আন্তরিকভাবে দুঃখিত। বিষয়টি দ্রুত সমাধানের জন্য অনুগ্রহ করে আমাদের বিস্তারিত জানান।",
                    f"আপনার এই সমস্যার জন্য আমরা আন্তরিকভাবে ক্ষমাপ্রার্থী। আমরা আপনার ক্ষোভ ও হতাশা বুঝতে পারছি। দয়া করে আপনার বিবরণ দিয়ে আমাদের সাথে যোগাযোগ করুন, আমরা এখনই সমাধানের ব্যবস্থা নিচ্ছি।",
                    f"{subject} নিয়ে যে ভোগান্তি হয়েছে তার জন্য আমরা গভীরভাবে দুঃখ প্রকাশ করছি। এই সমস্যাটি অবিলম্বে সমাধান করতে আমাদের টিম প্রস্তুত, অনুগ্রহ করে আমাদের ইনবক্সে মেসেজ দিন।",
                    f"পরিস্থিতিটা সত্যিই হতাশাজনক এবং এর জন্য আমরা দুঃখ প্রকাশ করছি। অনুগ্রহ করে আমাদের সাথে সরাসরি যোগাযোগ করুন যাতে দ্রুত এটি সমাধান করতে পারি।",
                ]
                return bn_neg[archetype_idx % len(bn_neg)]

            if category == "question_advice":
                bn_qa = [
                    f"দারুণ একটি প্রশ্ন! শুরুতে মূল বেসিক বিষয়গুলো শক্তভাবে আয়ত্ত করে ছোট ছোট প্রজেক্ট বানালে সবচেয়ে দ্রুত ও কার্যকরভাবে শিখতে পারবেন।",
                    f"আপনার মূল লক্ষ্য ও আগ্রহের ওপর নির্ভর করে সিদ্ধান্ত নেওয়া সবচেয়ে ভালো হবে। যেকোনো একটি দিকে স্থির থেকে এগিয়ে যান, সফলতা আসবেই।",
                    f"খুবই সময়োপযোগী প্রশ্ন! বাস্তবধর্মী কাজের মাধ্যমে শুরু করলে যেকোনো বিষয়ের ওপর দক্ষতা অর্জন অনেক সহজ হয়। শুভকামনা!",
                    f"চমৎকার প্রশ্ন! এটি শুরু করার পেছনে আপনার প্রধান পরিকল্পনা কী—কোন নির্দিষ্ট প্রজেক্ট নিয়ে কাজ করার ইচ্ছে আছে কি?",
                ]
                return bn_qa[archetype_idx % len(bn_qa)]

            if category == "food":
                bn_food = [
                    f"দেখেই তো জিভে জল চলে এলো! {subject}-এর অসাধারণ রূপ আর খাবারের আকর্ষণ সত্যিই দারুণ। 🍽️😋",
                    f"ঘরের তৈরি এমন সুস্বাদু {subject}-এর মজাই আলাদা! পরিবেশনটাও হয়েছে চমৎকার। শুভ ভোজন! ✨",
                    f"অসাধারণ দেখতে হয়েছে! এই {subject} রান্নার পেছনে আপনার কোনো বিশেষ সিক্রেট বা স্পেশাল মশলা আছে কি?",
                    f"একদম সেরা! দেখেই ক্ষুধা লেগে গেল। দারুণ রান্নার হাত আপনার! 💯🍽️",
                ]
                return bn_food[archetype_idx % len(bn_food)]

            if category == "pet":
                bn_pet = [
                    f"ইশ কী মিষ্টি আর মায়াবী চোখ! ওদের দেখলে সারাদিনের সব ক্লান্তি নিমেষেই হারিয়ে যায়। অনেক অনেক ভালোবাসা! 🐾❤️",
                    f"অসম্ভব কিউট! এমন আদুরে পোষা প্রাণী ঘরে থাকলে সবসময় আনন্দের পরিবেশ তৈরি হয়। ✨",
                    f"খুবই মিষ্টি! ওর বয়স কত এবং সবচেয়ে প্রিয় দুষ্টুমি কোনটা? 🐾",
                    f"মাশাল্লাহ, কী কিউট! অনেক অনেক আদর রইল এই ছোট্ট বন্ধুর জন্য। ❤️",
                ]
                return bn_pet[archetype_idx % len(bn_pet)]

            if category == "opinion":
                bn_opinion = [
                    f"খুবই বাস্তবধর্মী এবং চিন্তাশীল মতামত তুলে ধরেছেন! {subject} নিয়ে এমন গঠনমূলক আলোচনা সত্যিই ইতিবাচক।",
                    f"আপনার দৃষ্টিভঙ্গি খুবই যৌক্তিক। প্রতিটি বিষয়েরই নানা দিক থাকে, তবে আপনার যুক্তিগুলো বেশ জোরালো।",
                    f"দারুণ একটি পয়েন্ট! এই সিদ্ধান্তের পেছনে কোনো বিশেষ অভিজ্ঞতা বা কারণ কাজ করেছে কি?",
                ]
                return bn_opinion[archetype_idx % len(bn_opinion)]

            if category == "sports":
                bn_sports = [
                    f"{subject}-এর জন্য বিশাল অভিনন্দন! শেষ মুহূর্তের রোমাঞ্চ আর অনবদ্য পারফরম্যান্সে ম্যাচ জয় করে নিল দল। 🏆🔥",
                    f"চাপের মুখে এমন দুর্দান্ত খেলাই সত্যিকারের চ্যাম্পিয়নের পরিচয়! {subject}-এর ঐতিহাসিক জয়ে আমরা সবাই গর্বিত। 👏",
                    f"অসাধারণ এক ম্যাচ! পুরো খেলায় {subject}-এর কোন মুহূর্তটি আপনার কাছে সেরা মনে হয়েছে?",
                    f"সেরার সেরা পারফরম্যান্স! {subject} পুরো বাজিমাত করে দিল। 💯👏",
                ]
                return bn_sports[archetype_idx % len(bn_sports)]

            if category == "tech":
                bn_tech = [
                    f"{subject}-এর চমৎকার উদ্বোধনের জন্য অনেক অনেক অভিনন্দন! শুভকামনা রইল পুরো টিমের জন্য। 🚀✨",
                    f"দুর্দান্ত একটি উদ্ভাবন! {subject} অনেকের উপকারে আসবে। উদ্ভাবন ও একাগ্রতার দারুণ নিদর্শন। 👏",
                    f"দারুণ প্রজেক্ট! {subject}-এর পরবর্তী ভার্সনে কী কী নতুন ফিচার দেখার সুযোগ থাকবে?",
                    f"অসাধারণ কাজ! {subject} দিয়ে ডিজিটাল যাত্রা আরও সহজ ও ফলপ্রসূ হোক। 💯",
                ]
                return bn_tech[archetype_idx % len(bn_tech)]

            if category == "milestone":
                bn_milestone = [
                    f"{subject}-এর এই ঐতিহাসিক মাইলফলকে পৌঁছানোর জন্য আন্তরিক অভিনন্দন! 👏🏅",
                    f"অক্লান্ত পরিশ্রম আর একাগ্রতার সুন্দর ফলাফল {subject}। এগিয়ে যান বহুদূর! 🎉",
                    f"এমন বড় অর্জন সত্যিই অনুপ্রেরণাদায়ক! {subject}-এর সাথে জড়িত সবার জন্য শুভকামনা। 🙌",
                ]
                return bn_milestone[archetype_idx % len(bn_milestone)]

            if category == "creative":
                bn_creative = [
                    f"অনবদ্য দৃশ্য আর চমৎকার ফ্রেম! {subject}-এর নান্দনিক রূপ সত্যিই চোখে লেগে থাকার মতো। 🌸✨",
                    f"দারুণ ফটোগ্রাফি! এমন স্নিগ্ধ দৃশ্য মন ভালো করে দেয়। অসাধারণ শিল্পকর্ম!",
                ]
                return bn_creative[archetype_idx % len(bn_creative)]

            bn_general = [
                f"খুবই সুন্দর ও স্নিগ্ধ একটি মুহূর্ত! দিনটি আপনার দারুণ আর আনন্দময় কাটুক। 🌸✨",
                f"আমাদের সাথে {subject}-এর এই সুন্দর আপডেটটি শেয়ার করার জন্য অনেক ধন্যবাদ! পাশে আছি সবসময়। ❤️",
                f"দারুণ লাগলো দেখে! আপনার প্রতিটি দিন এমন আনন্দ আর ইতিবাচকতায় ভরে উঠুক। 🌟",
                f"চমৎকার একটি শেয়ার! সবসময় এমন হাসিখুশি ও প্রাণবন্ত থাকুন। ✨",
            ]
            return bn_general[archetype_idx % len(bn_general)]

        if sentiment == "negative" or category == "struggle":
            multi_neg = {
                "es": [
                    f"Lamentamos sinceramente esta experiencia tan frustrante con {subject}. Esto no refleja nuestro estándar. Por favor, envíenos un mensaje directo para resolverlo de inmediato.",
                    f"Sentimos mucho los inconvenientes con {subject}. Entendemos su frustración y queremos ayudarle a solucionarlo cuanto antes. Contáctenos directamente.",
                ],
                "fr": [
                    f"Nous vous présentons nos sincères excuses pour cette expérience frustrante avec {subject}. Veuillez nous contacter en message privé afin que nous puissions résoudre ce problème immédiatement.",
                    f"Nous regrettons profondément ce désagrément concernant {subject}. Nous comprenons votre frustration et souhaitons y remédier rapidement.",
                ],
                "de": [
                    f"Wir entschuldigen uns aufrichtig für diese frustrierende Erfahrung mit {subject}. Bitte senden Sie uns eine Direktnachricht, damit unser Team das Problem umgehend lösen kann.",
                    f"Es tut uns sehr leid zu hören, welche Unannehmlichkeiten {subject} verursacht hat. Wir möchten die Angelegenheit schnellstmöglich für Sie klären.",
                ],
                "pt": [
                    f"Pedimos sinceras desculpas por essa experiência frustrante com {subject}. Por favor, envie-nos uma mensagem direta para que possamos resolver isso imediatamente.",
                    f"Lamentamos muito o ocorrido com {subject}. Compreendemos sua frustração e queremos ajudá-lo a solucionar o problema o mais rápido possível.",
                ],
                "it": [
                    f"Ci scusiamo sinceramente per questa spiacevole e frustrante esperienza con {subject}. Ti preghiamo di scriverci in privato per risolvere subito la situazione.",
                    f"Siamo davvero spiacenti per l'inconveniente legato a {subject}. Comprendiamo la tua frustrazione e vogliamo risolvere al più presto.",
                ],
                "hi": [
                    f"{subject} को लेकर हुए इस निराशाजनक अनुभव के लिए हम ईमानदारी से क्षमा चाहते हैं। कृपया हमें सीधा संदेश भेजें ताकि हमारी टीम इसे तुरंत हल कर सके।",
                    f"हुई असुविधा के लिए हमें गहरा खेদ है। हम आपकी परेशानी को समझते हैं और तुरंत समाधान के लिए सहायता हेतु तैयार हैं।",
                ],
                "ar": [
                    f"نعتذر بصدق عن هذه التجربة المحبطة بخصوص {subject}. هذا ليس المستوى الذي نطمح إليه. يرجى مراسلتنا لحل المشكلة فوراً.",
                    f"نأسف بشدة للإزعاج الذي واجهته مع {subject}. نتفهم إحباطك ونرغب في مساعدتك لحل الأمر في أسرع وقت.",
                ],
                "zh": [
                    f"对于在{subject}上给您带来的不愉快经历，我们深表歉意。请直接私信我们，以便我们立即为您调查并解决该问题。",
                    f"非常抱歉给您带来困扰。我们完全理解您的感受，并希望能尽快为您解决问题，请随时与我们联系。",
                ],
                "ja": [
                    f"{subject}に関してご不便とご迷惑をおかけしましたことを心よりお詫び申し上げます。迅速に対応いたしますので、DMにて詳細をお知らせください。",
                    f"この度の不手際につき深くお詫び申し上げます。お客様の状況を真摯に受け止め、早急に解決へ向けて対応いたします。",
                ],
            }
            options = multi_neg.get(lang, multi_neg["es"])
            return options[archetype_idx % len(options)]

        multi_dict = {
            "es": [
                f"¡Muchas felicidades por {subject}! Un logro sumamente merecido y emocionante. 👏🏆",
                f"Qué gran momento para {subject}. Se nota todo el esfuerzo y la dedicación invertidos. ¡Enhorabuena!",
                f"¡Increíble desempeño con {subject}! Sin duda una demostración de auténtica perseverancia.",
            ],
            "fr": [
                f"Toutes mes félicitations pour {subject} ! Un moment magnifique et amplement mérité. 👏🎉",
                f"Quel accomplissement remarquable pour {subject} ! Bravo à toute l'équipe pour cette belle énergie.",
                f"Une performance vraiment inspirante avec {subject}. Merci beaucoup pour ce partage enrichissant !",
            ],
            "de": [
                f"Herzlichen Glückwunsch zu {subject}! Eine absolut herausragende und wohlverdiente Leistung. 👏🎉",
                f"Was für ein beeindruckender Moment für {subject}. Schön zu sehen, wie sich die harte Arbeit auszahlt!",
                f"Großartige Leistung mit {subject}. Das verdient vollsten Respekt und Anerkennung!",
            ],
            "pt": [
                f"Parabéns pelo sucesso em {subject}! Uma conquista incrível e muito merecida. 👏🔥",
                f"Que momento maravilhoso para {subject}! Dá para sentir a energia e a dedicação de todos.",
                f"Excelente resultado com {subject}. Desejo ainda mais success nas próximas etapas!",
            ],
            "it": [
                f"Congratulazioni vivissime per {subject}! Un traguardo fantastico e meritatissimo. 👏✨",
                f"Che momento straordinario per {subject}. Bellissimo vedere premiata tanta dedizione!",
            ],
            "hi": [
                f"{subject} पर बहुत-बहुत बधाई! यह वास्तव में एक शानदार और प्रेरणादायक उपलब्धि है। 👏🎉",
                f"{subject} के साथ यह प्रदर्शन देखकर बहुत खुशी हुई। आगे भी ऐसे ही आगे बढ़ते रहें!",
            ],
            "ar": [
                f"ألف مبروك على هذا الإنجاز الرائع في {subject}! نجاح مستحق بكل جدارة. 👏🎉",
                f"أداء مميز ومبهر في {subject}. نتمنى لكم دوام التوفيق والنجاح!",
            ],
            "zh": [
                f"热烈祝贺在{subject}上取得的优异成绩！这确实是一个实至名归的精彩时刻。👏🎉",
                f"非常精彩的表现！看到为{subject}付出的努力获得回报真是太棒了。",
            ],
            "ja": [
                f"{subject}での素晴らしい成果、本当におめでとうございます！努力が実を結びましたね。👏🎉",
                f"{subject}についての素晴らしい瞬間を共有していただきありがとうございます！",
            ],
        }
        options = multi_dict.get(lang, multi_dict["es"])
        return options[archetype_idx % len(options)]


comment_generator_model = BertCommentGeneratorModel()

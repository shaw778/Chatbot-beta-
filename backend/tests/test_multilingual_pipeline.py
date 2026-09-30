from backend.models.pipeline import comment_generator, full_pipeline
from backend.models.text_features import detect_language
from backend.models.toxicity_model import BertToxicityModel


def test_multilingual_pipeline_returns_structured_result():
    result = full_pipeline("¡Qué increíble experiencia! Gracias por compartir esto.")
    assert result["sentiment"]["bert"]["sentiment"] in {"positive", "negative", "neutral"}
    assert "schedule" in result
    assert "engagement" in result
    assert "toxicity" in result


def test_comment_generation_works_for_multilingual_input():
    result = full_pipeline("Bonjour, j'adore cette idée et je la trouve très utile.")
    assert "comment_generator" in result
    assert isinstance(result["comment_generator"]["comment"], str)
    assert len(result["comment_generator"]["comment"]) > 10
    assert "language" in result["comment_generator"]


def test_comment_generation_uses_input_topics():
    result = full_pipeline("The product launch was amazing and the community loved it.")
    comment = result["comment_generator"]["comment"].lower()
    assert any(keyword in comment for keyword in ["launch", "product", "community", "amazing"])


def test_multilingual_pipeline_bangla():
    post = "আজকের দিনটি সত্যিই অসাধারণ ছিল! নতুন প্রজেক্ট সফলভাবে উদ্বোধন করলাম।"
    result = full_pipeline(post)
    assert detect_language(post) == "bn"
    assert result["sentiment"]["bert"]["sentiment"] == "positive"
    assert result["toxicity"]["toxic"] is False

    cg = result["comment_generator"]
    assert cg["language"] == "bn"
    assert any("\u0980" <= c <= "\u09FF" for c in cg["comment"])


def test_bangla_facebook_comment_reply():
    prompt = (
        'Someone commented on a Facebook post. Write a short, friendly reply (1-2 sentences max).\n\n'
        'Original post: "আমাদের নতুন শীতকালীন কালেকশন এসে গেছে!"\n'
        'Comment: "ভাইয়া প্রোডাক্টটির দাম কত এবং কীভাবে অর্ডার করব?"\n\n'
        'Reply:'
    )
    result = comment_generator(prompt, language="bn")
    reply = result.get("comment", "")
    assert result.get("language") == "bn"
    assert any("\u0980" <= c <= "\u09FF" for c in reply)
    assert any(term in reply for term in ["ধন্যবাদ", "ইনবক্স", "অর্ডার", "বিস্তারিত"])


def test_bangla_toxicity_detection():
    tox_model = BertToxicityModel()
    clean_res = tox_model.predict("চমৎকার কাজ ভাইয়া, এগিয়ে যান শুভকামনা!")
    toxic_res = tox_model.predict("তুই একটা ফালতু বাটপার ও চিটার মানুষ, তোকে মেরে ফেলব")

    assert clean_res["toxic"] is False
    assert toxic_res["toxic"] is True
    assert any(term in toxic_res["keywords"] for term in ["মেরে ফেলব", "বাটপার", "চিটার", "ফালতু"])


def test_bangla_question_comment_generation():
    post = "নতুনদের জন্য কোন প্রোগ্রামিং ভাষা দিয়ে শুরু করলে সবচেয়ে ভালো হবে?"
    result = comment_generator(post, language="bn")
    comment = result.get("comment", "")
    assert result.get("language") == "bn"
    assert any("\u0980" <= c <= "\u09FF" for c in comment)
    assert any(w in comment for w in ["প্রশ্ন", "বেসিক", "প্রজেক্ট", "সিদ্ধান্ত", "লক্ষ্য", "শেখা", "দক্ষতা"])


def test_bangla_food_comment_generation():
    post = "আজকে রাতে পরিবারের জন্য স্পেশাল খাসির বিরিয়ানি রান্না করলাম।"
    result = comment_generator(post, language="bn")
    comment = result.get("comment", "")
    assert result.get("language") == "bn"
    assert any("\u0980" <= c <= "\u09FF" for c in comment)
    assert any(w in comment for w in ["জিভে জল", "খাবার", "মজা", "রান্না", "ভোজন", "সুস্বাদু", "ক্ষুধা"])


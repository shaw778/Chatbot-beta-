from backend.models.comment_generator_model import comment_generator_model
from backend.models.pipeline import comment_generator, full_pipeline, sentiment, toxicity
from backend.models.sentiment_model import sentiment_model
from backend.models.toxicity_model import toxicity_model


def test_sports_victory_sentiment_and_topics():
    post = "Desert Vipers claim victory in the DP World ILT20 after an incredible final over!"
    res = sentiment(post)
    assert res["sentiment"] == "positive"
    assert res["tone"] == "excited"
    assert any("Vipers" in t or "ILT20" in t or "victory" in t.lower() for t in res["topics"])
    assert res["confidence"] >= 0.75


def test_personal_insult_toxicity_flagged():
    insult_post = "You are an absolute clown and you suck at this"
    res = toxicity(insult_post)
    assert res["toxic"] is True
    assert res["label"] == "toxic"
    assert res["score"] >= 0.70
    assert any(k in res["keywords"] for k in ["clown", "you suck"])


def test_benign_negative_context_not_flagged_as_toxic():
    benign_post = "I hate when it rains on the weekend and ruins our plans"
    res = toxicity(benign_post)
    assert res["toxic"] is False
    assert res["label"] == "clean"
    assert res["score"] <= 0.20


def test_negation_flips_sentiment_correctly():
    # "not bad" should be positive
    res_pos = sentiment("Not bad at all, actually worked out pretty well.")
    assert res_pos["sentiment"] == "positive"

    # "not good" should be negative
    res_neg = sentiment("Not good, completely broken and disappointing.")
    assert res_neg["sentiment"] == "negative"


def test_comment_generator_diversity_for_sports():
    post = "Desert Vipers claim victory in the DP World ILT20 after an incredible final over!"
    res = comment_generator(post, sentiment_label="positive", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3

    # All 3 comments must be distinct
    assert len(set(comments)) == 3

    # Ensure none use the old rigid template
    for c in comments:
        assert "The part about" not in c
        assert "caught my attention" not in c
        assert len(c) > 15


def test_comment_generator_for_product_launch():
    post = "We just launched our new AI dashboard for developers!"
    res = comment_generator(post, sentiment_label="positive", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3
    assert len(set(comments)) == 3
    for c in comments:
        assert "The part about" not in c


def test_comment_generator_negative_empathy():
    post = "Flight got delayed by 6 hours and lost my luggage, worst travel day ever."
    res = comment_generator(post, sentiment_label="negative", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3
    # Check for professional apologetic customer service crisis resolution tone
    assert any(any(w in c.lower() for w in ["sorry", "apologize", "frustrating", "frustration", "resolve", "support"]) for c in comments)
    # Must NOT contain cheerful emojis
    for c in comments:
        assert "✨" not in c and "🔥" not in c and "💪" not in c and "☕" not in c


def test_comment_generator_negative_fb_reply():
    prompt = (
        'Someone commented on a Facebook post. Write a short reply (1-2 sentences max).\n\n'
        'Original post: "Our latest release is live."\n'
        'Comment: "This is broken and the worst experience ever, total scam."\n\n'
        'Reply:'
    )
    res = comment_generator(prompt, sentiment_label="negative", num_comments=1)
    comment = res.get("comment", "")
    assert isinstance(comment, str)
    assert any(w in comment.lower() for w in ["apologize", "sorry", "frustrat", "resolve", "right", "message"])
    assert "✨" not in comment and "🔥" not in comment


def test_comment_generator_fb_comment_reply():
    prompt = (
        'Someone commented on a Facebook post. Write a short, friendly reply (1-2 sentences max).\n\n'
        'Original post: "Championship win!"\n'
        'Comment: "Huge congratulations to the entire squad!"\n\n'
        'Reply:'
    )
    res = comment_generator(prompt, sentiment_label="neutral", num_comments=1)
    comment = res.get("comment", "")
    assert isinstance(comment, str)
    assert len(comment) > 10
    assert any(w in comment.lower() for w in ["thank", "appreciate", "celebrating", "support"])


def test_full_pipeline_cohesive_output():
    post = "Thrilled to announce we passed our final milestone today with flying colors!"
    result = full_pipeline(post)
    assert result["sentiment"]["bert"]["sentiment"] == "positive"
    assert result["toxicity"]["toxic"] is False
    assert "engagement" in result
    assert "schedule" in result
    assert "comment_generator" in result


def test_comment_generator_question_advice():
    post = "Should I learn Python or JavaScript first as a beginner?"
    res = comment_generator(post, sentiment_label="neutral", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3
    # Must not be an absurd congratulations
    for c in comments:
        assert "huge congratulations" not in c.lower()
        assert "love seeing this win" not in c.lower()
    # Should provide helpful perspective/encouragement
    assert any(any(w in c.lower() for w in ["fundamentals", "hands-on", "recommendation", "goals", "start", "projects", "journey"]) for c in comments)


def test_comment_generator_food():
    post = "Cooked homemade biryani for the family dinner tonight!"
    res = comment_generator(post, sentiment_label="positive", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3
    for c in comments:
        assert "huge congratulations" not in c.lower()
    assert any(any(w in c.lower() for w in ["mouthwatering", "delicious", "tasty", "recipe", "plate", "food", "chef", "comfort food"]) for c in comments)


def test_comment_generator_pets():
    post = "Our new puppy chewed on my favorite slippers today, look at this guilty face."
    res = comment_generator(post, sentiment_label="neutral", num_comments=3)
    comments = res.get("comments", [])
    assert len(comments) == 3
    for c in comments:
        assert "huge congratulations on slippers" not in c.lower()
    assert any(any(w in c.lower() for w in ["precious", "puppy", "pup", "cuteness", "adorable", "pets", "joy", "angel"]) for c in comments)


def test_benign_situational_frustration_toxicity():
    bug_post = "This stupid bug in the code is driving me crazy"
    res = toxicity(bug_post)
    assert res["toxic"] is False
    assert res["label"] == "clean"

    commute_post = "Traffic on my commute sucks today"
    res2 = toxicity(commute_post)
    assert res2["toxic"] is False
    assert res2["label"] == "clean"


def test_contrastive_sentiment_handling():
    contrast_post = "The design is gorgeous, but the customer service was awful and disappointing."
    res = sentiment(contrast_post)
    # The negative second clause dominates the critique
    assert res["sentiment"] == "negative"
    assert "mixed" in res["explanation"].lower() or "contrastive" in res["explanation"].lower()


from pathlib import Path
from backend.models.vision_model import local_vision_model


def test_local_vision_model_describes_image():
    test_img_path = Path(__file__).resolve().parents[2] / "frontend" / "test.png"
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    description = local_vision_model.describe_image(img_bytes)
    assert isinstance(description, str)
    assert len(description) > 5
    # Should describe flower or visual scene
    assert any(w in description.lower() for w in ["flower", "background", "image", "visual"])


def test_local_vision_model_generates_comment():
    description = "A black background with a white and red flower."
    comment = local_vision_model.generate_comment(description, tone="friendly")
    assert isinstance(comment, str)
    assert len(comment) > 10
    assert any(w in comment.lower() for w in ["flower", "bloom", "colors", "photo", "capture", "vivid"])


def test_local_vision_model_process_image_structure():
    test_img_path = Path(__file__).resolve().parents[2] / "frontend" / "test.png"
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    result = local_vision_model.process_image(img_bytes, tone="enthusiastic")
    assert "description" in result
    assert "comment" in result
    assert "text" in result
    assert "What's happening in the image:" in result["text"]
    assert "Generated Comment:" in result["text"]
    assert result["text"].index("What's happening in the image:") < result["text"].index("Generated Comment:")

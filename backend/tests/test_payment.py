import pytest
from unittest.mock import patch, MagicMock

from backend.database import database
from backend.payment_service import (
    BOT_PACKAGES,
    get_packages,
    initiate_payment,
    validate_and_activate,
    get_current_plan,
)


@pytest.fixture(autouse=True)
def setup_db():
    database.init()


def test_four_bot_packages_defined():
    """Verify that exactly 4 bot packages exist with full pricing and features."""
    packages = get_packages()
    assert len(packages) == 4
    expected_ids = ["starter", "creator_pro", "business_agency", "enterprise_unlimited"]
    for pkg_id in expected_ids:
        assert pkg_id in packages
        pkg = packages[pkg_id]
        assert "price_bdt" in pkg
        assert "price_usd" in pkg
        assert "features" in pkg
        assert len(pkg["features"]) >= 4
        assert pkg["price_bdt"] > 0

    assert packages["starter"]["price_bdt"] == 500
    assert packages["creator_pro"]["price_bdt"] == 1500
    assert packages["business_agency"]["price_bdt"] == 3500
    assert packages["enterprise_unlimited"]["price_bdt"] == 7500


def test_initiate_payment_creates_pending_record():
    """Verify initiate_payment creates a pending transaction in database."""
    fake_session = {
        "status": "SUCCESS",
        "sessionkey": "TEST_SESSION_KEY_12345",
        "GatewayPageURL": "https://sandbox.sslcommerz.com/EasyCheckOut/testcde12345",
    }

    with patch("backend.payment_service.get_sslcommerz_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.createSession.return_value = fake_session
        mock_client_factory.return_value = mock_client

        res = initiate_payment(
            package_id="creator_pro",
            customer={
                "cus_name": "Test User",
                "cus_email": "test@example.com",
                "cus_phone": "01711223344",
                "cus_city": "Dhaka",
            },
            base_url="http://127.0.0.1:8000",
        )

        assert res["success"] is True
        assert res["GatewayPageURL"] == fake_session["GatewayPageURL"]
        assert res["sessionkey"] == fake_session["sessionkey"]
        tran_id = res["tran_id"]
        assert tran_id.startswith("BOTPAY_")

        # Verify database record
        payment = database.get_payment(tran_id)
        assert payment is not None
        assert payment["status"] == "PENDING"
        assert payment["package_id"] == "creator_pro"
        assert float(payment["amount"]) == 1500.0


def test_ipn_validation_confirms_funds_and_unlocks_tier():
    """Verify that SSLCommerz IPN / validation confirms funds and unlocks bot tier."""
    import uuid
    test_tran_id = f"BOTPAY_TEST_VALIDATE_{uuid.uuid4().hex[:8]}"
    database.create_payment(
        tran_id=test_tran_id,
        package_id="creator_pro",
        package_name="Creator Bot (Pro)",
        amount=1500.0,
        currency="BDT",
        cus_name="Test User",
        cus_email="test@example.com",
        cus_phone="01711223344",
    )

    fake_validation_resp = {
        "status": "VALID",
        "tran_id": test_tran_id,
        "amount": "1500.00",
        "currency_amount": "1500.00",
        "currency": "BDT",
        "val_id": "VAL_ID_987654321",
        "card_type": "BKASH-BKash",
        "bank_tran_id": "BANK_TXN_ABCDEF",
    }

    with patch("backend.payment_service.get_sslcommerz_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.validationTransactionOrder.return_value = fake_validation_resp
        mock_client_factory.return_value = mock_client

        res = validate_and_activate(
            val_id="VAL_ID_987654321",
            tran_id=test_tran_id,
            raw_ipn_data={"card_type": "BKASH-BKash"},
        )

        assert res["success"] is True
        assert res["status"] == "VALID"
        assert res["tier"] == "Pro"

        # Check DB payment status updated
        payment = database.get_payment(test_tran_id)
        assert payment["status"] == "VALID"
        assert payment["val_id"] == "VAL_ID_987654321"

        # Check subscription unlocked
        active_plan = get_current_plan()
        assert active_plan["is_active"] is True
        assert active_plan["package_id"] == "creator_pro"
        assert active_plan["tier"] == "Pro"
        assert active_plan["comment_limit"] == 2500


def test_ipn_rejects_tampered_or_invalid_payment():
    """Verify that a tampered amount or invalid gateway status is rejected."""
    import uuid
    test_tran_id = f"BOTPAY_TEST_FAKE_{uuid.uuid4().hex[:8]}"
    database.create_payment(
        tran_id=test_tran_id,
        package_id="enterprise_unlimited",
        package_name="Enterprise Bot (Unlimited)",
        amount=7500.0,
        currency="BDT",
        cus_name="Attacker",
        cus_email="attacker@example.com",
        cus_phone="01700000000",
    )

    # Fake validation response: status is INVALID or amount mismatch (e.g. 50 BDT instead of 7500 BDT)
    fake_validation_resp = {
        "status": "VALID",
        "tran_id": test_tran_id,
        "amount": "50.00",  # Mismatched amount!
        "currency_amount": "50.00",
        "val_id": "FAKE_VAL_ID",
    }

    with patch("backend.payment_service.get_sslcommerz_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.validationTransactionOrder.return_value = fake_validation_resp
        mock_client_factory.return_value = mock_client

        res = validate_and_activate(val_id="FAKE_VAL_ID", tran_id=test_tran_id)
        assert res["success"] is False
        assert res["status"] == "INVALID"

        # Database payment must NOT be valid
        payment = database.get_payment(test_tran_id)
        assert payment["status"] == "INVALID"

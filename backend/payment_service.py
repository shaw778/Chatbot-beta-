import logging
import time
import uuid
from typing import Any, Dict, Optional

from sslcommerz_lib import SSLCOMMERZ

from .config import (
    APP_BASE_URL,
    SSLCOMMERZ_IS_SANDBOX,
    SSLCOMMERZ_STORE_ID,
    SSLCOMMERZ_STORE_PASSWD,
)
from .database import database

logger = logging.getLogger("chatbot.payment")

BOT_PACKAGES: Dict[str, Dict[str, Any]] = {
    "starter": {
        "id": "starter",
        "name": "Starter Bot",
        "tier": "Starter",
        "tag": "Essential",
        "price_bdt": 500,
        "price_usd": 5,
        "billing": "Monthly",
        "badge": "STARTER",
        "description": "Essential AI commenting for solo creators, personal brands, and small Facebook/Instagram pages.",
        "comment_limit": 500,
        "features": [
            "500 AI generated comments / month",
            "BERT Sentiment & Sarcasm analysis",
            "Real-time Toxicity & Safety filter",
            "1 Connected Social Media account",
            "Standard API response speed",
            "Saved Drafts & Manual Approvals",
        ],
        "is_popular": False,
    },
    "creator_pro": {
        "id": "creator_pro",
        "name": "Creator Bot (Pro)",
        "tier": "Pro",
        "tag": "Most Popular",
        "price_bdt": 1500,
        "price_usd": 15,
        "billing": "Monthly",
        "badge": "POPULAR",
        "description": "Visual intelligence and multi-channel posting automation for growing creators and influencers.",
        "comment_limit": 2500,
        "features": [
            "2,500 AI generated comments / month",
            "Image Comment Generation (Vision BLIP + AI)",
            "Smart Posting Scheduler (Random Forest + XGBoost)",
            "Facebook & Instagram Feed auto-reply",
            "Multi-tone & customizable persona",
            "Priority AI processing speed",
        ],
        "is_popular": True,
    },
    "business_agency": {
        "id": "business_agency",
        "name": "Business Bot (Agency)",
        "tier": "Business",
        "tag": "Best Value",
        "price_bdt": 3500,
        "price_usd": 35,
        "billing": "Monthly",
        "badge": "BEST VALUE",
        "description": "High-volume automation & team workflow for agencies and commercial multi-page brands.",
        "comment_limit": 10000,
        "features": [
            "10,000 AI generated comments / month",
            "Unlimited Facebook Pages & Instagram Profiles",
            "Batch comment generation & bulk scheduling",
            "Automated sentiment-driven reply engine",
            "Team approval workflow & audit logs",
            "Dedicated webhook & real-time alerts",
        ],
        "is_popular": False,
    },
    "enterprise_unlimited": {
        "id": "enterprise_unlimited",
        "name": "Enterprise Bot (Unlimited)",
        "tier": "Enterprise",
        "tag": "Unlimited VIP",
        "price_bdt": 7500,
        "price_usd": 75,
        "billing": "Monthly",
        "badge": "UNLIMITED",
        "description": "Custom fine-tuned models, dedicated infrastructure, and unlimited commenting scale.",
        "comment_limit": -1,  # unlimited
        "features": [
            "Unlimited AI comments & replies",
            "Dedicated fine-tuned BERT & LLM instance",
            "Custom brand guardrails & toxicity rules",
            "Instant Payment Notification (IPN) & custom hooks",
            "Zero rate-limit throttling",
            "24/7 Dedicated SLA & priority engineering support",
        ],
        "is_popular": False,
    },
}


def get_sslcommerz_client() -> SSLCOMMERZ:
    """Return an initialized SSLCOMMERZ SDK instance."""
    config = {
        "store_id": SSLCOMMERZ_STORE_ID or "testbox",
        "store_pass": SSLCOMMERZ_STORE_PASSWD or "qwerty",
        "issandbox": SSLCOMMERZ_IS_SANDBOX,
    }
    return SSLCOMMERZ(config)


def get_packages() -> Dict[str, Dict[str, Any]]:
    """Return available bot subscription packages."""
    return BOT_PACKAGES


def initiate_payment(
    package_id: str,
    customer: Dict[str, Any],
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Initiate an SSLCommerz payment session for a chosen bot package.
    Stores the pending payment in the database and returns the GatewayPageURL
    and sessionkey for Hosted Checkout and EasyCheckout (Seamless Modal).
    """
    package = BOT_PACKAGES.get(package_id)
    if not package:
        raise ValueError(f"Invalid package_id '{package_id}'. Choose from: {list(BOT_PACKAGES.keys())}")

    base = (base_url or APP_BASE_URL).rstrip("/")
    tran_id = f"BOTPAY_{int(time.time())}_{uuid.uuid4().hex[:6].upper()}"

    cus_name = str(customer.get("cus_name") or "Social Bot User").strip()
    cus_email = str(customer.get("cus_email") or "customer@example.com").strip()
    cus_phone = str(customer.get("cus_phone") or "01700000000").strip()
    cus_add1 = str(customer.get("cus_add1") or "Dhaka").strip()
    cus_city = str(customer.get("cus_city") or "Dhaka").strip()
    cus_country = str(customer.get("cus_country") or "Bangladesh").strip()

    # Record pending payment in database
    database.create_payment(
        tran_id=tran_id,
        package_id=package["id"],
        package_name=package["name"],
        amount=package["price_bdt"],
        currency="BDT",
        cus_name=cus_name,
        cus_email=cus_email,
        cus_phone=cus_phone,
    )

    client = get_sslcommerz_client()
    post_body = {
        "total_amount": package["price_bdt"],
        "currency": "BDT",
        "tran_id": tran_id,
        "success_url": f"{base}/payment/success",
        "fail_url": f"{base}/payment/fail",
        "cancel_url": f"{base}/payment/cancel",
        "ipn_url": f"{base}/payment/ipn",
        "emi_option": 0,
        "cus_name": cus_name,
        "cus_email": cus_email,
        "cus_phone": cus_phone,
        "cus_add1": cus_add1,
        "cus_city": cus_city,
        "cus_country": cus_country,
        "shipping_method": "NO",
        "num_of_item": 1,
        "product_name": package["name"],
        "product_category": "Software",
        "product_profile": "non-physical-goods",
    }

    try:
        response = client.createSession(post_body)
    except Exception as exc:
        logger.error("SSLCommerz createSession network error: %s", exc)
        raise RuntimeError(f"Could not connect to SSLCommerz gateway: {exc}") from exc

    status = response.get("status")
    if status == "SUCCESS":
        gateway_url = response.get("GatewayPageURL")
        sessionkey = response.get("sessionkey")
        logger.info("SSLCommerz session created successfully: tran_id=%s, sessionkey=%s", tran_id, sessionkey)
        return {
            "success": True,
            "tran_id": tran_id,
            "package": package,
            "GatewayPageURL": gateway_url,
            "sessionkey": sessionkey,
            "is_sandbox": SSLCOMMERZ_IS_SANDBOX,
        }
    else:
        failed_reason = response.get("failedreason") or str(response)
        logger.warning("SSLCommerz createSession failed: %s", failed_reason)
        # Update record to FAILED
        database.update_payment(tran_id, status="FAILED", validation_payload=response)
        return {
            "success": False,
            "tran_id": tran_id,
            "package": package,
            "error": f"Payment session initiation failed: {failed_reason}",
            "response": response,
        }


def validate_and_activate(
    val_id: str,
    tran_id: Optional[str] = None,
    raw_ipn_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate a payment with SSLCommerz gateway using val_id, confirm funds,
    and unlock bot premium features in database.
    """
    if not val_id:
        raise ValueError("val_id (Validation ID) is required for payment verification.")

    client = get_sslcommerz_client()

    try:
        val_resp = client.validationTransactionOrder(val_id)
    except Exception as exc:
        logger.error("SSLCommerz validationTransactionOrder API error: %s", exc)
        raise RuntimeError(f"SSLCommerz validation request failed: {exc}") from exc

    # Response is typically a dictionary or list
    if isinstance(val_resp, list) and val_resp:
        val_data = val_resp[0]
    elif isinstance(val_resp, dict):
        val_data = val_resp
    else:
        val_data = {}

    v_status = str(val_data.get("status", "")).upper()
    v_tran_id = val_data.get("tran_id") or tran_id
    v_amount = float(val_data.get("amount") or val_data.get("currency_amount") or 0.0)
    card_type = val_data.get("card_type") or (raw_ipn_data or {}).get("card_type")
    bank_tran_id = val_data.get("bank_tran_id") or (raw_ipn_data or {}).get("bank_tran_id")

    if not v_tran_id:
        return {"success": False, "error": "Missing transaction ID in validation response."}

    # Fetch stored transaction from database
    stored_payment = database.get_payment(v_tran_id)
    if not stored_payment:
        logger.warning("Validation received for unknown transaction: %s", v_tran_id)
        return {"success": False, "error": f"Transaction '{v_tran_id}' not found in database."}

    expected_amount = float(stored_payment["amount"])

    # Confirm funds and status
    is_valid = (v_status in {"VALID", "VALIDATED"}) and (abs(v_amount - expected_amount) < 0.01)

    if is_valid:
        # Update payment status
        database.update_payment(
            tran_id=v_tran_id,
            status="VALID",
            val_id=val_id,
            payment_method=card_type,
            bank_tran_id=bank_tran_id,
            validation_payload=val_data,
        )

        package_id = stored_payment["package_id"]
        package = BOT_PACKAGES.get(package_id, {
            "name": stored_payment["package_name"],
            "tier": "Pro",
        })

        # Activate subscription / Unlock premium bot features
        sub = database.activate_subscription(
            package_id=package_id,
            package_name=package["name"],
            tier=package.get("tier", "Pro"),
            tran_id=v_tran_id,
            duration_days=30,
        )

        logger.info(
            "Payment verified & bot tier activated: tran_id=%s, package=%s, val_id=%s",
            v_tran_id,
            package_id,
            val_id,
        )

        return {
            "success": True,
            "status": "VALID",
            "tran_id": v_tran_id,
            "val_id": val_id,
            "package_id": package_id,
            "package_name": package["name"],
            "tier": package.get("tier", "Pro"),
            "amount": v_amount,
            "subscription": sub,
            "validation_data": val_data,
        }
    else:
        fail_status = "INVALID"
        database.update_payment(
            tran_id=v_tran_id,
            status=fail_status,
            val_id=val_id,
            validation_payload=val_data,
        )
        logger.warning(
            "Payment verification failed: tran_id=%s, status=%s, received_amount=%s, expected_amount=%s",
            v_tran_id,
            v_status,
            v_amount,
            expected_amount,
        )
        return {
            "success": False,
            "status": fail_status,
            "tran_id": v_tran_id,
            "error": f"Gateway validation status was '{v_status}' or amount mismatch (expected {expected_amount}, got {v_amount}).",
            "validation_data": val_data,
        }


def get_current_plan() -> Dict[str, Any]:
    """Retrieve the current active bot plan and feature limits."""
    active_sub = database.get_active_subscription()
    if active_sub:
        package = BOT_PACKAGES.get(active_sub["package_id"], {})
        return {
            "is_active": True,
            "tier": active_sub["tier"],
            "package_id": active_sub["package_id"],
            "package_name": active_sub["package_name"],
            "comment_limit": package.get("comment_limit", 2500),
            "features": package.get("features", []),
            "activated_at": active_sub["activated_at"],
            "expires_at": active_sub["expires_at"],
            "tran_id": active_sub["tran_id"],
        }
    else:
        return {
            "is_active": False,
            "tier": "Free",
            "package_id": "free",
            "package_name": "Free Community Tier",
            "comment_limit": 50,
            "features": [
                "50 basic AI generated comments",
                "Basic VADER sentiment analysis",
                "Community support",
            ],
            "activated_at": None,
            "expires_at": None,
            "tran_id": None,
        }

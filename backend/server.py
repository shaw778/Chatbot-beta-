import os
import hashlib
import html
import re

import hmac
import io
import json
import logging
import threading
import time
import uuid
import base64
import urllib.error
import urllib.request
from pathlib import Path
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default
from urllib.parse import parse_qs, urlencode, urlsplit

from PIL import Image, ImageStat

from .ai_provider import call_ai, call_anthropic, call_openai, call_vision, fallback_response
from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    APP_BASE_URL,
    DB_ENGINE,
    FRONTEND_DIR,
    HOST,
    ROOT_DIR,
    META_APP_ID,
    META_APP_SECRET,
    META_REQUIRE_SIGNATURE,
    META_REDIRECT_URI,
    META_VERIFY_TOKEN,
    META_PAGE_ACCESS_TOKEN,
    META_PAGE_ID,
    META_LOGIN_CONFIG_ID,
    META_OAUTH_SCOPES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OAUTH_STATE_SECRET,
    PORT,
    RATE_LIMIT_IMAGE_REQUESTS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    SSLCOMMERZ_IS_SANDBOX,
    SSLCOMMERZ_STORE_ID,
    X_CLIENT_SECRET,
    X_CLIENT_ID,
    X_REDIRECT_URI,
)
from .database import database
from .models import pipeline
from .payment_service import (
    BOT_PACKAGES,
    get_current_plan,
    get_packages,
    initiate_payment,
    validate_and_activate,
)


logger = logging.getLogger("chatbot.server")
logging.basicConfig(level=logging.INFO, format="%(message)s")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 1 * 1024 * 1024
MIN_IMAGE_DIMENSION = 100
MIN_IMAGE_STDDEV = 5.0
PROMPT_INJECTION_TERMS = ("ignore previous", "ignore all instructions", "system prompt", "override instructions")
active_meta_page_access_token = META_PAGE_ACCESS_TOKEN
active_meta_page_id = META_PAGE_ID


class InMemoryRateLimiter:
    def __init__(self):
        self._events = {}
        self._lock = threading.Lock()

    def allow(self, key, limit, window_seconds):
        now = time.monotonic()
        with self._lock:
            recent = [timestamp for timestamp in self._events.get(key, []) if now - timestamp < window_seconds]
            allowed = len(recent) < limit
            if allowed:
                recent.append(now)
            self._events[key] = recent
            return allowed, max(0, int(window_seconds - (now - recent[0]))) if recent else 0


rate_limiter = InMemoryRateLimiter()


def read_json(handler):
    raw = read_body(handler, MAX_JSON_BYTES).decode("utf-8")
    return json.loads(raw or "{}")


def read_request_data(handler):
    """Safely parse body as JSON or Form URL-encoded according to Content-Type."""
    ct = handler.headers.get("Content-Type", "").lower()
    body = read_body(handler, MAX_JSON_BYTES)
    if "json" in ct:
        try:
            return json.loads(body.decode("utf-8") or "{}")
        except Exception:
            return {}
    if "application/x-www-form-urlencoded" in ct:
        raw_text = body.decode("utf-8", errors="replace")
        parsed = parse_qs(raw_text)
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
    # Fallback try json then form
    try:
        return json.loads(body.decode("utf-8") or "{}")
    except Exception:
        raw_text = body.decode("utf-8", errors="replace")
        parsed = parse_qs(raw_text)
        return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()} if parsed else {}


def get_request_base_url(handler):
    host = handler.headers.get("Host")
    if host:
        scheme = handler.headers.get("X-Forwarded-Proto", "http")
        return f"{scheme}://{host}"
    return APP_BASE_URL


def redirect_to_payment_result(handler, status, tran_id=None, package_id=None, error=None):
    params = {"payment": status}
    if tran_id:
        params["tran_id"] = tran_id
    if package_id:
        params["plan"] = package_id
    if error:
        params["payment_error"] = str(error)[:240]
    location = "/?" + urlencode(params)
    handler.send_response(HTTPStatus.FOUND)
    handler.send_header("Location", location)
    handler.send_header("X-Request-ID", getattr(handler, "request_id", uuid.uuid4().hex[:16]))
    handler.end_headers()


def send_json(handler, payload, status=HTTPStatus.OK, headers=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("X-Request-ID", getattr(handler, "request_id", "unknown"))
    for name, value in (headers or {}).items():
        handler.send_header(name, str(value))
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_body(handler, max_bytes=None):
    length = int(handler.headers.get("Content-Length", "0"))
    if max_bytes is not None and length > max_bytes:
        raise ValueError(f"Request exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return handler.rfile.read(length)


def parse_multipart(body, content_type):
    if "boundary=" not in content_type.lower():
        raise ValueError("Multipart boundary is missing.")
    message = BytesParser(policy=default).parsebytes(
        (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode("utf-8") + body
    )
    fields = {}
    files = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        value = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            files[name] = {"filename": filename, "content_type": part.get_content_type(), "data": value}
        else:
            fields[name] = value.decode("utf-8", errors="replace")
    return fields, files


def inspect_image(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            grayscale = image.convert("L")
            stddev = ImageStat.Stat(grayscale).stddev[0]
            width, height = image.size
            image_format = image.format or "unknown"
    except Exception as exc:
        raise ValueError("Image processing rejected. The uploaded file is not a readable image.") from exc

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION or stddev < MIN_IMAGE_STDDEV:
        raise ValueError("Image processing rejected. Asset is blank, severely blurred, or too small.")
    return {"width": width, "height": height, "format": image_format, "pixel_stddev": round(stddev, 2)}


def has_prompt_injection(text):
    normalized = " ".join(text.lower().split())
    return any(term in normalized for term in PROMPT_INJECTION_TERMS)


def verify_meta_signature(handler, body):
    if not META_APP_SECRET:
        return not META_REQUIRE_SIGNATURE
    header = handler.headers.get("X-Hub-Signature-256", "")
    if not header.startswith("sha256="):
        return False
    supplied = header[7:]
    expected = hmac.new(META_APP_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(supplied, expected)


def send_meta_message(recipient_id, text):
    if not active_meta_page_access_token:
        raise RuntimeError("META_PAGE_ACCESS_TOKEN is not configured.")
    payload = json.dumps({
        "recipient": {"id": recipient_id},
        "message": {"text": text[:2000]},
    }).encode("utf-8")
    query = urlencode({"access_token": active_meta_page_access_token})
    request = urllib.request.Request(
        "https://graph.facebook.com/v22.0/me/messages?" + query,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Meta Graph API HTTP {exc.code}: {error[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Meta Graph API connection error: {exc.reason}") from exc


def publish_meta_page_post(message, scheduled_at=None):
    if not active_meta_page_access_token or not active_meta_page_id:
        raise RuntimeError("Facebook Page access is not configured. Connect a Page or set META_PAGE_ID and META_PAGE_ACCESS_TOKEN.")
    fields = {"message": message}
    if scheduled_at is not None:
        fields.update({"published": "false", "scheduled_publish_time": str(int(scheduled_at / 1000))})
    query = urlencode({"access_token": active_meta_page_access_token})
    request = urllib.request.Request(
        f"https://graph.facebook.com/v22.0/{active_meta_page_id}/feed?{query}",
        data=urlencode(fields).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Facebook Page post failed with HTTP {exc.code}: {error[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Facebook Page post connection error: {exc.reason}") from exc


def resolve_fb_share_url(url):
    """Follow HTTP redirect for Facebook /share/ or shortened links to find canonical URL."""
    if "/share/" not in url and "fb.watch" not in url:
        return url
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    headers = {
        "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        opener.open(req, timeout=6)
        return url
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            return exc.headers.get("Location") or url
    except Exception as exc:
        logger.info("Redirect resolution error for %s: %s", url, exc)
        return url
    return url


def scrape_public_fb_post(url):
    """Scrape public Facebook post metadata (og:description / og:title) via crawler."""
    headers = {
        "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=7) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
            og_desc = re.search(r'property="og:description"\s+content="([^"]+)"', content) or re.search(r'content="([^"]+)"\s+property="og:description"', content)
            og_title = re.search(r'property="og:title"\s+content="([^"]+)"', content) or re.search(r'content="([^"]+)"\s+property="og:title"', content)
            meta_desc = re.search(r'name="description"\s+content="([^"]+)"', content) or re.search(r'content="([^"]+)"\s+name="description"', content)

            text = ""
            if og_desc:
                text = og_desc.group(1)
            elif meta_desc:
                text = meta_desc.group(1)
            elif og_title:
                text = og_title.group(1)

            title = og_title.group(1) if og_title else ""
            if text:
                return {
                    "message": html.unescape(text),
                    "title": html.unescape(title),
                }
    except Exception as exc:
        logger.info("Scraper failed on %s: %s", url, exc)
    return None


def extract_fb_post_id(url_or_id):
    """Extract a Graph API post ID from any Facebook URL or raw ID."""
    text = url_or_id.strip().strip("\"'<>[]()")
    if re.fullmatch(r"\d+_\d+", text):
        return text
    if re.fullmatch(r"pfbid[A-Za-z0-9]+", text):
        return text
    if re.fullmatch(r"\d+", text):
        return f"{active_meta_page_id}_{text}" if active_meta_page_id and len(text) > 10 else text

    # If it's a URL, resolve redirects (such as /share/p/...) first
    if "http" in text:
        text = resolve_fb_share_url(text)

    # Check for pfbid anywhere in the text
    pfbid_match = re.search(r"(pfbid[A-Za-z0-9]+)", text)
    if pfbid_match:
        pfbid = pfbid_match.group(1)
        if "?" in text:
            q = parse_qs(urlsplit(text).query)
            pid = q.get("id", [""])[0]
            if pid and pid.isdigit():
                return f"{pid}_{pfbid}"
        return pfbid

    if not re.match(r"^[a-z][a-z0-9+.-]*://", text, re.IGNORECASE):
        text = "https://" + text
    parsed = urlsplit(text)
    hostname = (parsed.hostname or "").lower()
    if not any(hostname == domain or hostname.endswith("." + domain) for domain in ("facebook.com", "fb.com", "fb.watch")):
        raise ValueError("Enter a Facebook post URL or post ID.")

    query = parse_qs(parsed.query)
    story_fbid = query.get("story_fbid", [""])[0]
    page_id = query.get("id", [""])[0] or active_meta_page_id
    if story_fbid:
        if page_id and "_" not in story_fbid:
            return f"{page_id}_{story_fbid}"
        return story_fbid

    fbid = query.get("fbid", [""])[0]
    if fbid:
        if page_id and "_" not in fbid and fbid != page_id:
            return f"{page_id}_{fbid}"
        return fbid

    video_v = query.get("v", [""])[0]
    if video_v:
        return video_v

    post_param = query.get("post_id", [""])[0]
    if post_param:
        return post_param

    path = parsed.path.rstrip("/")
    m = re.search(r"/(\d+)/posts/(\d+)", path)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    m = re.search(r"/groups/(\d+)/(?:posts|permalink)/(\d+)", path)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    m = re.search(r"/posts/(\d+)", path)
    if m:
        post_id = m.group(1)
        return f"{active_meta_page_id}_{post_id}" if active_meta_page_id and "_" not in post_id else post_id

    m = re.search(r"/(?:photos|photo)/.*?(\d+)", path)
    if m:
        photo_id = m.group(1)
        return f"{active_meta_page_id}_{photo_id}" if active_meta_page_id else photo_id

    m = re.search(r"/(?:reel|reels|videos|video|permalink)/(\d+)", path)
    if m:
        return m.group(1)

    m = re.search(r"/share/[pvr]/([A-Za-z0-9_]+)", path)
    if m:
        share_id = m.group(1)
        if share_id.isdigit() and active_meta_page_id:
            return f"{active_meta_page_id}_{share_id}"
        return share_id

    # Fallback: any sequence of 10+ digits in the path
    m = re.findall(r"\d{10,}", path)
    if m:
        candidate = m[-1]
        return f"{active_meta_page_id}_{candidate}" if active_meta_page_id and "_" not in candidate else candidate

    raise ValueError("Could not identify a post ID in that Facebook link. Copy the post permalink and try again.")


def fetch_facebook_post(url_or_id):
    """Fetch a Facebook post's content via fast-path scraping for external URLs and Graph API for page posts."""
    if not active_meta_page_access_token:
        raise RuntimeError("Facebook Page access token is not configured.")

    raw = str(url_or_id).strip()
    is_http = "http://" in raw or "https://" in raw
    is_page_post = bool(active_meta_page_id and active_meta_page_id in raw)

    # 1. Fast path for external URLs: scrape public metadata first (~1.5s vs 15s Graph API rejection)
    if is_http and not is_page_post:
        scraped = scrape_public_fb_post(raw)
        if scraped and scraped.get("message"):
            try:
                resolved_id = extract_fb_post_id(raw)
            except Exception:
                resolved_id = "fb_post"
            return {
                "id": resolved_id,
                "message": scraped["message"],
                "name": scraped.get("title", ""),
                "permalink_url": raw,
                "is_external": True,
            }

    # 2. Resolve share links if needed
    target_url = raw
    if is_http:
        target_url = resolve_fb_share_url(raw)

    # 3. Extract resolved post ID
    resolved_id = extract_fb_post_id(target_url)

    # 4. Attempt Graph API
    candidates = [resolved_id]
    if active_meta_page_id and "_" not in resolved_id and not resolved_id.startswith("pfbid"):
        candidates.append(f"{active_meta_page_id}_{resolved_id}")
    elif active_meta_page_id and "_" in resolved_id:
        prefix, suffix = resolved_id.split("_", 1)
        if prefix != active_meta_page_id:
            candidates.append(f"{active_meta_page_id}_{suffix}")

    def request_candidate(pid, fields):
        query = urlencode({"fields": fields, "access_token": active_meta_page_access_token})
        req = urllib.request.Request(
            f"https://graph.facebook.com/v22.0/{pid}?{query}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                return json.loads(resp.read().decode("utf-8")), None
        except urllib.error.HTTPError as exc:
            return None, (exc.code, exc.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, socket.timeout, Exception) as exc:
            logger.warning("Graph API candidate %s request error: %s", pid, exc)
            return None, (408, f"Facebook request timed out: {exc}")

    last_error = None
    for cand in candidates:
        post, err = request_candidate(cand, "id,message,story,created_time,permalink_url")
        if post and (post.get("message") or post.get("story")):
            post["is_external"] = not (active_meta_page_id and str(post.get("id", "")).startswith(str(active_meta_page_id)))
            if not post.get("permalink_url") and is_http:
                post["permalink_url"] = target_url
            return post
        if err and "nonexisting field (message)" in err[1]:
            post, err = request_candidate(cand, "id,name,created_time")
            if post:
                post["is_external"] = not (active_meta_page_id and str(post.get("id", "")).startswith(str(active_meta_page_id)))
                if not post.get("permalink_url") and is_http:
                    post["permalink_url"] = target_url
                return post
        last_error = err

    # 5. Fallback: If Graph API denied access and scraper wasn't called yet
    if is_http:
        scraped = scrape_public_fb_post(target_url)
        if scraped and scraped.get("message"):
            return {
                "id": resolved_id,
                "message": scraped["message"],
                "name": scraped.get("title", ""),
                "permalink_url": target_url,
                "is_external": True,
            }

    if last_error:
        code, msg = last_error
        if code == 408 or "timed out" in str(msg).lower():
            raise RuntimeError("Connection to Facebook timed out. Please check your internet connection or enter the post text directly below.")
        raise RuntimeError(f"Could not fetch Facebook post (HTTP {code}): {str(msg)[:300]}")
    raise RuntimeError("Could not fetch Facebook post. Check the post URL or ID.")


def post_facebook_comment(post_id, message):
    """Post a comment on a Facebook post with candidate fallbacks."""
    if not active_meta_page_access_token:
        raise RuntimeError("Facebook Page access token is not configured.")
    resolved_id = extract_fb_post_id(post_id)

    candidates = [resolved_id]
    if active_meta_page_id and "_" not in resolved_id and not resolved_id.startswith("pfbid"):
        candidates.append(f"{active_meta_page_id}_{resolved_id}")
    elif active_meta_page_id and "_" in resolved_id:
        prefix, suffix = resolved_id.split("_", 1)
        if prefix != active_meta_page_id:
            candidates.append(f"{active_meta_page_id}_{suffix}")

    last_error = None
    for cand in candidates:
        query = urlencode({"access_token": active_meta_page_access_token})
        payload = urlencode({"message": message}).encode("utf-8")
        req = urllib.request.Request(
            f"https://graph.facebook.com/v22.0/{cand}/comments?{query}",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Facebook comment connection error: {exc.reason}") from exc

    err_str = last_error or "Unknown error"
    if "singular statuses API" in err_str or "#12" in err_str or "200" in err_str or "sufficient permissions" in err_str or "does not exist" in err_str or "missing permissions" in err_str or "Page Public Content Access" in err_str or "Unsupported" in err_str:
        raise RuntimeError(
            "Meta Graph API only permits direct automated commenting on posts that belong to your own connected Page. "
            "For third-party public posts (like this one), Meta blocks API commenting. "
            "Please use the '📋 Copy & Open on FB' button to paste your generated comment directly on the post, "
            "or click '📢 Post to My Page Feed' to publish it to your Page!"
        )
    raise RuntimeError(f"Facebook comment failed: {err_str[:300]}")


def reply_to_facebook_comment(comment_id, message):
    """Reply to a specific comment on Facebook."""
    if not active_meta_page_access_token:
        raise RuntimeError("Facebook Page access token is not configured.")
    query = urlencode({"access_token": active_meta_page_access_token})
    payload = urlencode({"message": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://graph.facebook.com/v22.0/{comment_id}/comments?{query}",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Facebook reply failed (HTTP {exc.code}): {error[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Facebook reply connection error: {exc.reason}") from exc


def fetch_page_feed_posts(limit=10):
    """Fetch recent Page posts and comments safely without failing completely on permission boundaries."""
    if not active_meta_page_access_token or not active_meta_page_id:
        raise RuntimeError("Facebook Page is not connected.")

    # 1. Fetch posts using /posts or /published_posts (which succeeds with pages_manage_posts)
    posts = []
    last_err = None
    for endpoint in ("posts", "published_posts", "feed"):
        query = urlencode({
            "fields": "id,message,story,created_time,permalink_url",
            "limit": str(limit),
            "access_token": active_meta_page_access_token,
        })
        request = urllib.request.Request(
            f"https://graph.facebook.com/v22.0/{active_meta_page_id}/{endpoint}?{query}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                posts = json.loads(response.read().decode("utf-8")).get("data", [])
                break
        except urllib.error.HTTPError as exc:
            last_err = exc.read().decode("utf-8", errors="replace")
            logger.info("Endpoint /%s denied for posts: %s", endpoint, last_err[:150])
        except (urllib.error.URLError, TimeoutError, socket.timeout, Exception) as exc:
            last_err = f"Facebook connection error: {exc}"
            logger.warning("Endpoint /%s network error: %s", endpoint, exc)
            continue

    if not posts and last_err:
        if "timed out" in str(last_err).lower():
            raise RuntimeError("Facebook connection timed out. Please check your internet connection and try again.")
        raise RuntimeError(f"Could not load Page feed: {last_err[:200]}")

    # 2. For each post, attempt to retrieve comments; if pages_read_user_content is missing, provide empty comment list
    for post in posts:
        post["comments"] = {"data": []}
        post_id = post.get("id")
        if not post_id:
            continue
        c_query = urlencode({
            "fields": "id,message,from,created_time",
            "limit": "5",
            "access_token": active_meta_page_access_token,
        })
        c_req = urllib.request.Request(
            f"https://graph.facebook.com/v22.0/{post_id}/comments?{c_query}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(c_req, timeout=8) as c_resp:
                c_data = json.loads(c_resp.read().decode("utf-8")).get("data", [])
                post["comments"] = {"data": c_data}
        except Exception:
            # If reading comments is blocked due to pages_read_user_content, keep empty list
            pass

    return posts


def handle_meta_messages(payload):
    processed = 0
    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            message = event.get("message") or {}
            sender_id = (event.get("sender") or {}).get("id")
            text = str(message.get("text", "")).strip()
            if not sender_id or not text or message.get("is_echo"):
                continue
            result = call_ai({
                "provider": "anthropic",
                "system": "You are a concise, friendly Facebook Page chatbot. Reply directly to the user's message in at most two sentences.",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 180,
            })
            reply = str(result.get("text", "")).strip()
            if not reply:
                continue
            send_meta_message(sender_id, reply)
            database.store_conversation("facebook-messenger", text, reply)
            processed += 1
    return processed


def oauth_state(provider):
    nonce = uuid.uuid4().hex
    value = f"{provider}:{nonce}"
    if not OAUTH_STATE_SECRET:
        return ""
    signature = hmac.new(OAUTH_STATE_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{value}:{signature}"


def valid_oauth_state(provider, state):
    if not OAUTH_STATE_SECRET or not state:
        return False
    parts = state.split(":")
    if len(parts) != 3 or parts[0] != provider:
        return False
    value = ":".join(parts[:2])
    expected = hmac.new(OAUTH_STATE_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(parts[2], expected)


def exchange_oauth_code(provider, code):
    if provider == "meta":
        query = urlencode({
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "redirect_uri": META_REDIRECT_URI,
            "code": code,
        })
        request = urllib.request.Request(
            "https://graph.facebook.com/v22.0/oauth/access_token?" + query,
            headers={"Accept": "application/json"},
        )
    elif provider == "x":
        payload = urlencode({
            "code": code,
            "grant_type": "authorization_code",
            "client_id": X_CLIENT_ID,
            "redirect_uri": X_REDIRECT_URI,
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.twitter.com/2/oauth2/token",
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Authorization": "Basic " + base64.b64encode(
                    f"{X_CLIENT_ID}:{X_CLIENT_SECRET}".encode("utf-8")
                ).decode("ascii"),
            },
            method="POST",
        )
    else:
        raise ValueError("Unsupported social provider.")

    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not data.get("access_token"):
        raise RuntimeError("The social platform did not return an access token.")
    return data


def exchange_meta_page_token(user_access_token):
    query = urlencode({"fields": "id,name,access_token,tasks", "access_token": user_access_token})
    request = urllib.request.Request(
        "https://graph.facebook.com/v22.0/me/accounts?" + query,
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            pages = json.loads(response.read().decode("utf-8")).get("data", [])
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Meta Page lookup failed with HTTP {exc.code}: {error[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Meta Page lookup connection error: {exc.reason}") from exc
    if not pages:
        raise RuntimeError(
            "No Facebook Pages found for this account. "
            "Make sure you granted pages_show_list, pages_manage_posts, "
            "and pages_messaging permissions, and that your Meta App is "
            "in Live mode or the user is listed as a tester/developer."
        )
    for page in pages:
        if page.get("access_token"):
            logger.info("Connected to Facebook Page: %s (ID %s)", page.get("name", "?"), page.get("id"))
            return page
    raise RuntimeError(
        "Facebook returned pages but none included a usable Page access token. "
        "Re-authorize and make sure you grant all requested permissions."
    )


def persist_env_vars(env_path, updates):
    env_path = Path(env_path)
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    existing_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            k = k.strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                existing_keys.add(k)
                continue
        new_lines.append(line)
    for k, v in updates.items():
        if k not in existing_keys:
            new_lines.append(f"{k}={v}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def redirect_to_connection_result(handler, provider, error=None):
    params = {"connected": provider} if not error else {"connect_error": error[:240]}
    location = "/?" + urlencode(params)
    handler.send_response(HTTPStatus.FOUND)
    handler.send_header("Location", location)
    handler.send_header("X-Request-ID", handler.request_id)
    handler.end_headers()


class ChatbotHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = urlsplit(path).path
        if path == "/":
            return str(FRONTEND_DIR / "index.html")
        return str(FRONTEND_DIR / path.lstrip("/"))

    def log_message(self, format, *args):
        logger.info(json.dumps({
            "event": "http_request",
            "request_id": getattr(self, "request_id", "unknown"),
            "client": self.client_address[0],
            "message": format % args,
        }))

    def _begin_request(self):
        self.request_id = uuid.uuid4().hex[:16]

    def _check_rate_limit(self, path):
        limit = RATE_LIMIT_IMAGE_REQUESTS if path == "/api/comment/image" else RATE_LIMIT_REQUESTS
        allowed, retry_after = rate_limiter.allow(
            f"{self.client_address[0]}:{path}", limit, RATE_LIMIT_WINDOW_SECONDS
        )
        if allowed:
            return True
        send_json(
            self,
            {"error": "Rate limit exceeded. Please try again shortly."},
            HTTPStatus.TOO_MANY_REQUESTS,
            {"Retry-After": max(1, retry_after)},
        )
        return False

    def do_GET(self):
        global active_meta_page_access_token, active_meta_page_id
        self._begin_request()
        path = urlsplit(self.path).path
        if path == "/api/health":
            return send_json(
                self,
                {
                    "ok": True,
                    "database_engine": DB_ENGINE,
                    "database": database.label(),
                    "anthropic_key_configured": bool(ANTHROPIC_API_KEY),
                    "openai_key_configured": bool(OPENAI_API_KEY),
                    "meta_page_token_configured": bool(active_meta_page_access_token),
                    "meta_page_id_configured": bool(active_meta_page_id),
                    "meta_page_id": active_meta_page_id or "",
                    "meta_app_id_configured": bool(META_APP_ID),
                    "anthropic_model": ANTHROPIC_MODEL,
                    "openai_model": OPENAI_MODEL,
                },
            )

        if path == "/api/conversations":
            return send_json(self, {"items": database.recent_conversations()})

        if path == "/api/schedules":
            return send_json(self, {"items": database.scheduled_comments()})

        if path == "/api/payment/packages":
            return send_json(self, {"packages": get_packages(), "active_plan": get_current_plan()})

        if path == "/api/payment/status":
            return send_json(self, {
                "active_plan": get_current_plan(),
                "recent_payments": database.list_payments(limit=10),
                "sslcommerz_mode": "sandbox" if SSLCOMMERZ_IS_SANDBOX else "live",
                "store_id": SSLCOMMERZ_STORE_ID,
            })

        if path == "/api/payment/history":
            return send_json(self, {"payments": database.list_payments(limit=50)})

        if path in {"/payment/success", "/payment/fail", "/payment/cancel"}:
            query = parse_qs(urlsplit(self.path).query)
            data = {k: v[0] for k, v in query.items()}
            tran_id = data.get("tran_id")
            val_id = data.get("val_id")
            if path == "/payment/success":
                if val_id:
                    v_res = validate_and_activate(val_id, tran_id=tran_id, raw_ipn_data=data)
                    return redirect_to_payment_result(
                        self,
                        "success" if v_res.get("success") else "failed",
                        tran_id=tran_id,
                        package_id=v_res.get("package_id"),
                        error=v_res.get("error"),
                    )
                return redirect_to_payment_result(self, "success", tran_id=tran_id)
            elif path == "/payment/fail":
                if tran_id:
                    database.update_payment(tran_id, status="FAILED", validation_payload=data)
                return redirect_to_payment_result(self, "failed", tran_id=tran_id, error=data.get("failedreason") or "Payment failed")
            elif path == "/payment/cancel":
                if tran_id:
                    database.update_payment(tran_id, status="CANCELLED", validation_payload=data)
                return redirect_to_payment_result(self, "cancelled", tran_id=tran_id, error="Payment cancelled")

        if path.startswith("/auth/") and path.endswith("/start"):
            provider = path.split("/")[2]
            if provider in {"meta", "instagram"}:
                if not META_APP_ID or not OAUTH_STATE_SECRET:
                    return redirect_to_connection_result(
                        self,
                        "meta",
                        "Meta OAuth is not configured. Add META_APP_ID and OAUTH_STATE_SECRET to .env, then restart the server.",
                    )
                state = oauth_state("meta")
                params = {
                    "client_id": META_APP_ID,
                    "redirect_uri": META_REDIRECT_URI,
                    "state": state,
                    "response_type": "code",
                    "scope": META_OAUTH_SCOPES,
                }
                if META_LOGIN_CONFIG_ID:
                    params["config_id"] = META_LOGIN_CONFIG_ID
                location = "https://www.facebook.com/v22.0/dialog/oauth?" + urlencode(params)
            elif provider == "x":
                if not X_CLIENT_ID or not OAUTH_STATE_SECRET:
                    return redirect_to_connection_result(
                        self,
                        "x",
                        "X OAuth is not configured. Add X_CLIENT_ID and OAUTH_STATE_SECRET to .env, then restart the server.",
                    )
                state = oauth_state("x")
                params = {"response_type": "code", "client_id": X_CLIENT_ID, "redirect_uri": X_REDIRECT_URI, "scope": "tweet.read users.read offline.access", "state": state}
                location = "https://twitter.com/i/oauth2/authorize?" + urlencode(params)
            else:
                return send_json(self, {"error": "Unsupported social provider."}, HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.send_header("X-Request-ID", self.request_id)
            self.end_headers()
            return

        if path.startswith("/auth/") and path.endswith("/callback"):
            provider = path.split("/")[2]
            query = parse_qs(urlsplit(self.path).query)
            state = query.get("state", [""])[0]
            if provider == "instagram":
                provider = "meta"
            if provider not in {"meta", "x"} or not valid_oauth_state(provider, state):
                return redirect_to_connection_result(self, provider, "OAuth state validation failed.")
            if query.get("error"):
                error_reason = query.get("error_reason", [""])[0]
                error_desc = query.get("error_description", ["Authorization was declined."])[0]
                logger.warning("OAuth denied for %s: %s — %s", provider, error_reason, error_desc)
                return redirect_to_connection_result(self, provider, error_desc or "Authorization was declined.")
            code = query.get("code", [""])[0]
            if not code:
                return redirect_to_connection_result(self, provider, "The provider did not return an authorization code.")
            try:
                token_data = exchange_oauth_code(provider, code)
                if provider == "meta":
                    page = exchange_meta_page_token(token_data["access_token"])
                    active_meta_page_access_token = page["access_token"]
                    active_meta_page_id = page["id"]
                    try:
                        persist_env_vars(ROOT_DIR / ".env", {
                            "META_PAGE_ACCESS_TOKEN": active_meta_page_access_token,
                            "META_PAGE_ID": active_meta_page_id,
                        })
                    except Exception as err:
                        logger.warning("Could not persist page credentials to .env: %s", err)
                    logger.info("Facebook Page connected: id=%s", active_meta_page_id)
            except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                logger.warning("OAuth token exchange failed for %s: %s", provider, exc)
                return redirect_to_connection_result(self, provider, str(exc)[:240])
            return redirect_to_connection_result(self, provider)

        if path == "/api/facebook/webhook":
            query = parse_qs(urlsplit(self.path).query)
            mode = query.get("hub.mode", [""])[0]
            token = query.get("hub.verify_token", [""])[0]
            challenge = query.get("hub.challenge", [""])[0]
            if META_VERIFY_TOKEN and mode == "subscribe" and hmac.compare_digest(token, META_VERIFY_TOKEN):
                body = challenge.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Request-ID", self.request_id)
                self.end_headers()
                self.wfile.write(body)
                return
            return send_json(self, {"error": "Webhook verification failed."}, HTTPStatus.FORBIDDEN)

        return super().do_GET()

    def do_POST(self):
        self._begin_request()
        path = urlsplit(self.path).path
        if not self._check_rate_limit(path):
            return
        try:
            if path in {"/initiate-payment", "/api/payment/initiate"}:
                payload = read_request_data(self)
                package_id = payload.get("package_id")
                if not package_id:
                    return send_json(self, {"error": "package_id is required."}, HTTPStatus.BAD_REQUEST)
                customer = {
                    "cus_name": payload.get("cus_name") or payload.get("name") or "Social Bot User",
                    "cus_email": payload.get("cus_email") or payload.get("email") or "customer@example.com",
                    "cus_phone": payload.get("cus_phone") or payload.get("phone") or "01700000000",
                    "cus_add1": payload.get("cus_add1") or payload.get("address") or "Dhaka",
                    "cus_city": payload.get("cus_city") or payload.get("city") or "Dhaka",
                    "cus_country": payload.get("cus_country") or "Bangladesh",
                }
                base_url = get_request_base_url(self)
                result = initiate_payment(package_id, customer, base_url)
                return send_json(self, result, HTTPStatus.OK if result.get("success") else HTTPStatus.BAD_REQUEST)

            if path in {"/payment/ipn", "/api/payment/ipn"}:
                data = read_request_data(self)
                val_id = data.get("val_id")
                tran_id = data.get("tran_id")
                logger.info("SSLCommerz IPN received: val_id=%s, tran_id=%s", val_id, tran_id)
                if not val_id:
                    return send_json(self, {"error": "Missing val_id in IPN notification."}, HTTPStatus.BAD_REQUEST)
                res = validate_and_activate(val_id, tran_id=tran_id, raw_ipn_data=data)
                if res.get("success"):
                    return send_json(self, {"status": "SUCCESS", "message": "Transaction verified and unlocked", "data": res})
                else:
                    return send_json(self, {"status": "FAILED", "error": res.get("error")}, HTTPStatus.BAD_REQUEST)

            if path in {"/payment/success", "/api/payment/success"}:
                data = read_request_data(self)
                val_id = data.get("val_id")
                tran_id = data.get("tran_id")
                logger.info("Payment success callback received: val_id=%s, tran_id=%s", val_id, tran_id)
                if val_id:
                    res = validate_and_activate(val_id, tran_id=tran_id, raw_ipn_data=data)
                    if res.get("success"):
                        return redirect_to_payment_result(self, "success", tran_id=tran_id, package_id=res.get("package_id"))
                    else:
                        return redirect_to_payment_result(self, "failed", tran_id=tran_id, error=res.get("error"))
                if tran_id:
                    stored = database.get_payment(tran_id)
                    if stored and stored.get("status") == "VALID":
                        return redirect_to_payment_result(self, "success", tran_id=tran_id, package_id=stored.get("package_id"))
                return redirect_to_payment_result(self, "success", tran_id=tran_id)

            if path in {"/payment/fail", "/api/payment/fail"}:
                data = read_request_data(self)
                tran_id = data.get("tran_id")
                if tran_id:
                    database.update_payment(tran_id, status="FAILED", validation_payload=data)
                return redirect_to_payment_result(self, "failed", tran_id=tran_id, error=data.get("failedreason") or "Payment failed or declined.")

            if path in {"/payment/cancel", "/api/payment/cancel"}:
                data = read_request_data(self)
                tran_id = data.get("tran_id")
                if tran_id:
                    database.update_payment(tran_id, status="CANCELLED", validation_payload=data)
                return redirect_to_payment_result(self, "cancelled", tran_id=tran_id, error="Payment was cancelled by the user.")

            if path == "/api/payment/validate":
                payload = read_json(self)
                val_id = payload.get("val_id")
                tran_id = payload.get("tran_id")
                if not val_id:
                    return send_json(self, {"error": "val_id is required."}, HTTPStatus.BAD_REQUEST)
                res = validate_and_activate(val_id, tran_id=tran_id)
                return send_json(self, res)

            if path == "/api/payment/simulate-test":
                payload = read_json(self)
                tran_id = payload.get("tran_id")
                stored = database.get_payment(tran_id)
                if not stored:
                    return send_json(self, {"error": f"Transaction '{tran_id}' not found."}, HTTPStatus.NOT_FOUND)
                sim_val = f"SIM_VAL_{int(time.time())}_{uuid.uuid4().hex[:6].upper()}"
                database.update_payment(
                    tran_id=tran_id,
                    status="VALID",
                    val_id=sim_val,
                    payment_method="SSL_BKASH_TEST",
                    bank_tran_id=f"BANK_{uuid.uuid4().hex[:8].upper()}",
                    validation_payload={"status": "VALID", "simulated": True, "amount": stored["amount"]},
                )
                pkg = BOT_PACKAGES.get(stored["package_id"], {})
                sub = database.activate_subscription(
                    package_id=stored["package_id"],
                    package_name=stored["package_name"],
                    tier=pkg.get("tier", "Pro"),
                    tran_id=tran_id,
                )
                return send_json(self, {
                    "success": True,
                    "status": "VALID",
                    "tran_id": tran_id,
                    "val_id": sim_val,
                    "subscription": sub,
                    "message": "Simulated sandbox payment verified and plan unlocked!",
                })

            if path == "/api/comment/image":
                return self.handle_image_comment()

            if path == "/api/facebook/webhook":
                body = read_body(self, 1 * 1024 * 1024)
                if not verify_meta_signature(self, body):
                    return send_json(self, {"error": "Signature verification failed."}, HTTPStatus.UNAUTHORIZED)
                payload = json.loads(body.decode("utf-8"))
                if payload.get("object") != "page":
                    return send_json(self, {"status": "IGNORED_OBJECT"})
                processed = handle_meta_messages(payload)
                return send_json(self, {"status": "EVENT_RECEIVED", "processed_messages": processed})

            if path == "/api/facebook/post":
                payload = read_json(self)
                message = str(payload.get("message", "")).strip()
                scheduled_at = payload.get("scheduled_at")
                if not message:
                    return send_json(self, {"error": "A Facebook post message is required."}, HTTPStatus.BAD_REQUEST)
                if scheduled_at is not None and (not isinstance(scheduled_at, int) or scheduled_at <= int(time.time() * 1000)):
                    return send_json(self, {"error": "A future schedule time is required."}, HTTPStatus.BAD_REQUEST)
                result = publish_meta_page_post(message, scheduled_at)
                return send_json(self, {"success": True, "post": result}, HTTPStatus.CREATED)

            if path == "/api/facebook/fetch-post":
                payload = read_json(self)
                url = str(payload.get("url", "")).strip()
                if not url:
                    return send_json(self, {"error": "A Facebook post URL or ID is required."}, HTTPStatus.BAD_REQUEST)
                post_data = fetch_facebook_post(url)
                return send_json(self, {"success": True, "post": post_data})

            if path == "/api/facebook/comment":
                payload = read_json(self)
                post_id = str(payload.get("post_id", "")).strip()
                message = str(payload.get("message", "")).strip()
                if not post_id or not message:
                    return send_json(self, {"error": "Both post_id and message are required."}, HTTPStatus.BAD_REQUEST)
                try:
                    result = post_facebook_comment(post_id, message)
                    database.store_conversation("facebook-comment", message, json.dumps(result))
                    return send_json(self, {"success": True, "comment": result}, HTTPStatus.CREATED)
                except RuntimeError as exc:
                    err_msg = str(exc)
                    is_ext = ("third-party" in err_msg.lower() or "external" in err_msg.lower() or "meta graph api only permits" in err_msg.lower())
                    return send_json(self, {
                        "error": err_msg,
                        "is_external": is_ext
                    }, HTTPStatus.BAD_REQUEST)

            if path == "/api/facebook/feed":
                payload = read_json(self)
                limit = int(payload.get("limit", 10))
                posts = fetch_page_feed_posts(min(limit, 25))
                return send_json(self, {"success": True, "posts": posts})

            if path == "/api/facebook/reply":
                payload = read_json(self)
                comment_id = str(payload.get("comment_id", "")).strip()
                message = str(payload.get("message", "")).strip()
                if not comment_id or not message:
                    return send_json(self, {"error": "Both comment_id and message are required."}, HTTPStatus.BAD_REQUEST)
                result = reply_to_facebook_comment(comment_id, message)
                database.store_conversation("facebook-reply", message, json.dumps(result))
                return send_json(self, {"success": True, "reply": result}, HTTPStatus.CREATED)

            if path == "/api/schedules":
                payload = read_json(self)
                comment = str(payload.get("comment", "")).strip()
                platform = str(payload.get("platform", "")).strip().lower()
                scheduled_at = payload.get("scheduled_at")
                if not comment:
                    return send_json(self, {"error": "A comment is required."}, HTTPStatus.BAD_REQUEST)
                if platform not in {"facebook", "instagram", "twitter"}:
                    return send_json(self, {"error": "Choose Facebook, Instagram, or Twitter/X."}, HTTPStatus.BAD_REQUEST)
                if not isinstance(scheduled_at, int) or scheduled_at <= 0:
                    return send_json(self, {"error": "A valid schedule date and time is required."}, HTTPStatus.BAD_REQUEST)
                return send_json(self, {"item": database.create_scheduled_comment(comment, platform, scheduled_at)}, HTTPStatus.CREATED)

            if path.startswith("/api/models/"):
                return self.handle_model_endpoint()

            if path == "/api/ai":
                payload = read_json(self)
                mode = payload.pop("mode", "chat")
                result = call_ai(payload)
                user_text = ""
                messages = payload.get("messages") or []
                if messages:
                    user_text = str(messages[-1].get("content", ""))
                database.store_conversation(mode, user_text, result.get("text", ""))
                return send_json(self, result)

            if path == "/api/claude":
                payload = read_json(self)
                mode = payload.pop("mode", "chat")
                result = call_ai({**payload, "provider": "anthropic"})
                user_text = ""
                messages = payload.get("messages") or []
                if messages:
                    user_text = str(messages[-1].get("content", ""))
                database.store_conversation(mode, user_text, result.get("text", ""))
                return send_json(self, result)

            if path == "/api/chatgpt":
                payload = read_json(self)
                mode = payload.pop("mode", "chat")
                result = call_ai({**payload, "provider": "openai"})
                user_text = ""
                messages = payload.get("messages") or []
                if messages:
                    user_text = str(messages[-1].get("content", ""))
                database.store_conversation(mode, user_text, result.get("text", ""))
                return send_json(self, result)

            return send_json(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            return send_json(self, {"error": "Request body must contain valid JSON."}, HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE if "exceeds" in str(exc).lower() else HTTPStatus.BAD_REQUEST
            return send_json(self, {"error": str(exc)}, status)
        except RuntimeError as exc:
            return send_json(self, {"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception:
            logger.exception("Unhandled request failure")
            return send_json(self, {"error": "Internal server error."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_image_comment(self):
        content_type = self.headers.get("Content-Type", "")
        try:
            body = read_body(self, MAX_IMAGE_BYTES + 1024 * 1024)
            fields, files = parse_multipart(body, content_type)
            image = files.get("image_file")
            if not image or not image["data"]:
                return send_json(self, {"error": "image_file is required."}, HTTPStatus.BAD_REQUEST)
            if len(image["data"]) > MAX_IMAGE_BYTES:
                return send_json(self, {"error": "Image file exceeds the 10 MB limit."}, HTTPStatus.UNPROCESSABLE_ENTITY)
            if not image["content_type"].startswith("image/"):
                return send_json(self, {"error": "Uploaded asset must be an image."}, HTTPStatus.BAD_REQUEST)
            metadata = inspect_image(image["data"])
            prompt = fields.get("prompt", "")
            if has_prompt_injection(prompt) or has_prompt_injection(image["filename"]):
                logger.warning("Prompt-injection text was rejected in an image request.")
                return send_json(self, {"error": "Image instructions are treated as data, not commands."}, HTTPStatus.UNPROCESSABLE_ENTITY)

            tone = fields.get("tone", "friendly").strip() or "friendly"
            platform = fields.get("platform", "facebook").strip() or "facebook"
            provider = fields.get("provider", "openai")
            vision_prompt = (
                f"Analyze this image and write a {tone} social-media comment for {platform}.\n"
                "Step 1: In 1-2 clear sentences, describe what is actually happening in the image (subject, setting, actions, mood).\n"
                f"Step 2: Write an engaging, natural {tone} comment suitable for {platform}.\n\n"
                "Format your output exactly as:\n"
                "Description: [What is happening in the image]\n"
                "Comment: [Generated social media comment]"
            )
            try:
                result = call_vision(image["data"], image["content_type"], vision_prompt, provider=provider)
            except Exception as exc:
                logger.warning("Image vision provider unavailable: %s", exc)
                from .models.vision_model import local_vision_model
                result = local_vision_model.process_image(image["data"], tone=tone, platform=platform)

            description = result.get("description", "")
            comment = result.get("comment", "")
            raw_text = str(result.get("text", "")).strip()

            if not description and not comment:
                if "Description:" in raw_text and "Comment:" in raw_text:
                    parts = raw_text.split("Comment:", 1)
                    description = parts[0].replace("Description:", "").strip()
                    comment = parts[1].strip()
                elif "What's happening in the image:" in raw_text and "Generated Comment:" in raw_text:
                    parts = raw_text.split("Generated Comment:", 1)
                    description = parts[0].replace("What's happening in the image:", "").replace("🖼️", "").strip()
                    comment = parts[1].replace("💬", "").strip()
                else:
                    from .models.vision_model import local_vision_model
                    description = local_vision_model.describe_image(image["data"])
                    comment = raw_text

            if not description:
                from .models.vision_model import local_vision_model
                description = local_vision_model.describe_image(image["data"])

            if not comment:
                from .models.vision_model import local_vision_model
                comment = local_vision_model.generate_comment(description, tone=tone, platform=platform)

            full_text = f"🖼️ What's happening in the image:\n{description}\n\n💬 Generated Comment:\n{comment}"
            database.store_conversation("image-comment", image["filename"], full_text)
            return send_json(self, {
                "success": True,
                "filename": image["filename"],
                "image": metadata,
                "description": description,
                "comment": comment,
                "full_text": full_text,
            })
        except ValueError as exc:
            return send_json(self, {"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except json.JSONDecodeError:
            return send_json(self, {"error": "Webhook payload must be valid JSON."}, HTTPStatus.BAD_REQUEST)

    def handle_model_endpoint(self):
        payload = read_json(self)
        text = payload.get("text", "")
        sentiment_label = payload.get("sentiment", "neutral")
        provider = payload.get("provider", "anthropic")

        if self.path == "/api/models/sentiment":
            result = pipeline.sentiment(text, provider=provider)
            database.store_model_prediction("sentiment", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/toxicity":
            result = pipeline.toxicity(text, provider=provider)
            database.store_model_prediction("toxicity", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/engagement":
            result = pipeline.engagement(text, sentiment_label, provider=provider)
            database.store_model_prediction("engagement", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/schedule":
            result = pipeline.schedule(text, sentiment_label, payload.get("predicted_engagement"), provider=provider)
            database.store_model_prediction("schedule", text, result, result.get("source"))
            return send_json(self, result)

        if self.path == "/api/models/pipeline":
            result = pipeline.full_pipeline(text, provider=provider)
            database.store_model_prediction("full_pipeline", text, result, "mixed")
            return send_json(self, result)

        if self.path == "/api/models/comment-generator":
            style = payload.get("style", "casual")
            platform = payload.get("platform", "general")
            num_comments = payload.get("num_comments", payload.get("numComments", 1))
            language = payload.get("language")
            result = pipeline.comment_generator(text, sentiment_label, payload.get("toxicity", False), style=style, provider=provider, platform=platform, num_comments=num_comments, language=language)
            database.store_model_prediction("comment_generator", text, result, result.get("source"))
            return send_json(self, result)

        return send_json(self, {"error": "Model endpoint not found"}, HTTPStatus.NOT_FOUND)


class WSGIHandler(ChatbotHandler):
    """Bridge WSGI requests to ChatbotHandler for Gunicorn deployment."""

    def __init__(self, environ, start_response):
        self.environ = environ
        self.start_response = start_response
        self.headers_set = []
        self.status_code = 200
        self.status_message = "OK"
        self.wfile = io.BytesIO()

        content_length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(content_length) if content_length > 0 else b""
        self.rfile = io.BytesIO(body)

        path = environ.get("PATH_INFO", "/")
        query = environ.get("QUERY_STRING", "")
        self.path = f"{path}?{query}" if query else path
        self.command = environ.get("REQUEST_METHOD", "GET")
        self.request_version = environ.get("SERVER_PROTOCOL", "HTTP/1.1")
        self.client_address = (environ.get("REMOTE_ADDR", "127.0.0.1"), int(environ.get("REMOTE_PORT", 0) or 0))
        self.server = None
        self.close_connection = False

        from email.message import Message

        self.headers = Message()
        if environ.get("CONTENT_TYPE"):
            self.headers["Content-Type"] = environ["CONTENT_TYPE"]
        if environ.get("CONTENT_LENGTH"):
            self.headers["Content-Length"] = environ["CONTENT_LENGTH"]
        for k, v in environ.items():
            if k.startswith("HTTP_"):
                hdr = k[5:].replace("_", "-").title()
                self.headers[hdr] = v

    def send_response(self, code, message=None):
        self.status_code = code
        self.status_message = message or (HTTPStatus(code).phrase if code in HTTPStatus else "OK")

    def send_header(self, keyword, value):
        self.headers_set.append((keyword, str(value)))

    def end_headers(self):
        pass

    def log_message(self, format, *args):
        logger.info("%s - - [%s] %s", self.client_address[0], time.strftime("%d/%b/%Y %H:%M:%S"), format % args)


def application(environ, start_response):
    database.init()
    handler = WSGIHandler(environ, start_response)
    method = handler.command.upper()
    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()
    elif method == "HEAD":
        handler.do_HEAD()
    else:
        handler.send_error(HTTPStatus.NOT_IMPLEMENTED, "Unsupported method")

    status_str = f"{handler.status_code} {handler.status_message}"
    start_response(status_str, handler.headers_set)
    return [handler.wfile.getvalue()]


app = application


def main():
    database.init()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    server = ThreadingHTTPServer((host, port), ChatbotHandler)
    print(f"Chatbot running at http://{host}:{port}")
    print(f"Database engine: {DB_ENGINE}")
    print(f"Database: {database.label()}")

    if port != 8000 and not os.environ.get("RENDER"):
        try:
            alt_server = ThreadingHTTPServer((host, 8000), ChatbotHandler)
            threading.Thread(target=alt_server.serve_forever, daemon=True).start()
            print(f"Also listening at http://{host}:8000 for backward compatibility")
        except OSError:
            pass

    server.serve_forever()


if __name__ == "__main__":
    main()


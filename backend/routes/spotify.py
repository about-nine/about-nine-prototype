# routes/spotify.py - Spotify auth, token, and search endpoints
import base64
import os
import secrets
import sys
import time
from urllib.parse import urlencode

import requests
from flask import Blueprint, jsonify, redirect, request, session

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SPOTIFY_MARKET = os.getenv("SPOTIFY_MARKET", "US")

SPOTIFY_SCOPES = "streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state"
SPOTIFY_SEARCH_LIMIT = 10
SPOTIFY_SCAN_PAGES = 5

spotify_bp = Blueprint("spotify", __name__, url_prefix="/api/spotify")
spotify_auth_bp = Blueprint("spotify_auth", __name__)

_spotify_app_token = None
_spotify_app_expires_at = 0


def log(msg):
    """Force stdout flush for immediate logging"""
    print(msg, flush=True)
    sys.stdout.flush()


def spotify_config_ready():
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REDIRECT_URI)


def exchange_code_for_token(code: str):
    auth = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    encoded = base64.b64encode(auth.encode("utf-8")).decode("utf-8")
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        headers={"Authorization": f"Basic {encoded}"},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def refresh_access_token(refresh_token: str):
    auth = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    encoded = base64.b64encode(auth.encode("utf-8")).decode("utf-8")
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Authorization": f"Basic {encoded}"},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


def get_app_access_token():
    global _spotify_app_token, _spotify_app_expires_at

    if _spotify_app_token and time.time() < _spotify_app_expires_at - 30:
        log(f"🔄 Using cached app token (expires in {int(_spotify_app_expires_at - time.time())}s)")
        return _spotify_app_token

    log("🔑 Requesting new app access token...")
    log(f"   CLIENT_ID: {SPOTIFY_CLIENT_ID[:10]}..." if SPOTIFY_CLIENT_ID else "   CLIENT_ID: NOT SET!")
    log(f"   CLIENT_SECRET: {'*' * 10}..." if SPOTIFY_CLIENT_SECRET else "   CLIENT_SECRET: NOT SET!")

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        log("❌ Missing Spotify credentials!")
        return None

    try:
        auth = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        encoded = base64.b64encode(auth.encode("utf-8")).decode("utf-8")
        
        log("📡 POST https://accounts.spotify.com/api/token")
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {encoded}"},
            timeout=5,
        )
        
        log(f"📡 Token response: {r.status_code}")
        
        if r.status_code != 200:
            log(f"❌ Token request failed: {r.status_code}")
            log(f"📄 Response: {r.text}")
            return None
        
        data = r.json()
        _spotify_app_token = data.get("access_token")
        expires_in = data.get("expires_in", 3600)
        _spotify_app_expires_at = time.time() + int(expires_in)
        
        log(f"✅ Got app token (expires in {expires_in}s)")
        log(f"   Token preview: {_spotify_app_token[:20]}..." if _spotify_app_token else "   Token: NONE")
        
        return _spotify_app_token
    
    except Exception as e:
        log(f"❌ Exception getting app token: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


def normalize_limit(value):
    try:
        requested_limit = int(value)
    except (TypeError, ValueError):
        requested_limit = SPOTIFY_SEARCH_LIMIT
    return max(1, min(requested_limit, SPOTIFY_SEARCH_LIMIT))


def normalize_offset(value):
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def search_tracks(q, limit=None, offset=None):
    if not q:
        return []

    token = get_app_access_token()
    if not token:
        raise RuntimeError("SPOTIFY_CLIENT_ID/SECRET missing")

    limit_value = normalize_limit(limit)
    offset_value = normalize_offset(offset)

    params = {
        "q": q,
        "type": "track",
        "limit": limit_value,
        "market": SPOTIFY_MARKET,
    }
    if offset_value is not None:
        params["offset"] = offset_value

    items = []
    total = None
    current_offset = params.get("offset", 0) or 0
    pages = 0

    while len(items) < limit_value and pages < SPOTIFY_SCAN_PAGES:
        r = requests.get(
            "https://api.spotify.com/v1/search",
            params={
                **params,
                "limit": SPOTIFY_SEARCH_LIMIT,
                "offset": current_offset,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )

        r.raise_for_status()
        data = r.json().get("tracks", {})
        page_items = data.get("items", [])
        if total is None:
            total = data.get("total")

        if not page_items:
            break

        items.extend(page_items)
        pages += 1
        current_offset += len(page_items)
        if total is not None and current_offset >= total:
            break

    tracks = []
    fallback_tracks = []
    seen_ids = set()

    for it in items:
        preview_url = it.get("preview_url")
        album_images = it.get("album", {}).get("images", [])
        image_url = album_images[0]["url"] if album_images else ""
        artists = ", ".join([a.get("name", "") for a in it.get("artists", [])])
        track_id = it.get("id")

        track = {
            "id": track_id,
            "uri": it.get("uri"),
            "name": it.get("name"),
            "artist": artists,
            "image": image_url,
            "preview_url": preview_url,
            "has_preview": bool(preview_url),
            "duration_ms": it.get("duration_ms"),
        }

        if track_id and track_id in seen_ids:
            continue
        if track_id:
            seen_ids.add(track_id)

        if preview_url:
            tracks.append(track)
        else:
            fallback_tracks.append(track)

        if len(tracks) >= limit_value:
            break

    if len(tracks) < limit_value:
        for track in fallback_tracks:
            if len(tracks) >= limit_value:
                break
            tracks.append(track)

    return tracks


def _search_response():
    q = request.args.get("q")

    if not q:
        return jsonify(success=True, tracks=[])

    try:
        tracks = search_tracks(
            q,
            limit=request.args.get("limit"),
            offset=request.args.get("offset"),
        )
    except RuntimeError as e:
        return jsonify(success=False, message=str(e)), 500
    except Exception as e:
        details = ""
        if hasattr(e, "response") and e.response is not None:
            details = e.response.text
        print("🔥 Spotify API error:", e, details)
        return jsonify(success=True, tracks=[])

    return jsonify(success=True, tracks=tracks)


def get_valid_access_token():
    refresh_token = session.get("spotify_refresh_token")
    if not refresh_token:
        return None, "not connected"

    expires_at = int(session.get("spotify_expires_at") or 0)
    access_token = session.get("spotify_access_token")

    if access_token and time.time() < expires_at - 30:
        return access_token, None

    try:
        token_data = refresh_access_token(refresh_token)
    except Exception:
        return None, "token refresh failed"

    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 3600)
    new_refresh_token = token_data.get("refresh_token")

    if not access_token:
        return None, "token refresh failed"

    session["spotify_access_token"] = access_token
    session["spotify_expires_at"] = int(time.time()) + int(expires_in)
    if new_refresh_token:
        session["spotify_refresh_token"] = new_refresh_token

    return access_token, None


@spotify_bp.route("/login")
def spotify_login():
    if not spotify_config_ready():
        return jsonify(success=False, message="spotify config missing"), 500

    state = secrets.token_urlsafe(16)
    session["spotify_state"] = state
    session["spotify_next"] = request.args.get("next") or "/lounge.html"

    params = {
        "response_type": "code",
        "client_id": SPOTIFY_CLIENT_ID,
        "scope": SPOTIFY_SCOPES,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "state": state,
        "show_dialog": "true",
    }

    auth_url = "https://accounts.spotify.com/authorize?" + urlencode(params)
    return redirect(auth_url)


@spotify_bp.route("/search")
def spotify_search():
    return _search_response()


@spotify_auth_bp.route("/spotify/callback")
def spotify_callback():
    error = request.args.get("error")
    if error:
        return jsonify(success=False, message=error), 400

    state = request.args.get("state")
    if not state or state != session.get("spotify_state"):
        return jsonify(success=False, message="invalid state"), 400

    code = request.args.get("code")
    if not code:
        return jsonify(success=False, message="missing code"), 400

    try:
        token_data = exchange_code_for_token(code)
    except Exception as e:
        return jsonify(success=False, message="token exchange failed"), 400

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in", 3600)

    if not access_token or not refresh_token:
        return jsonify(success=False, message="invalid token response"), 400

    session["spotify_access_token"] = access_token
    session["spotify_refresh_token"] = refresh_token
    session["spotify_expires_at"] = int(time.time()) + int(expires_in)

    next_url = session.pop("spotify_next", "/lounge.html")
    session.pop("spotify_state", None)
    return redirect(next_url)


@spotify_bp.route("/status")
def spotify_status():
    connected = bool(session.get("spotify_refresh_token"))
    return jsonify(success=True, connected=connected)


@spotify_bp.route("/logout", methods=["POST"])
def spotify_logout():
    session.pop("spotify_access_token", None)
    session.pop("spotify_refresh_token", None)
    session.pop("spotify_expires_at", None)
    return jsonify(success=True)


@spotify_bp.route("/token")
def spotify_token():
    access_token, error = get_valid_access_token()
    if error:
        return jsonify(success=False, message=error), 401
    return jsonify(
        success=True,
        access_token=access_token,
        expires_at=session.get("spotify_expires_at"),
    )

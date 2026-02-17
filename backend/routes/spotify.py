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

SPOTIFY_SCOPES = "streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state"

spotify_bp = Blueprint("spotify", __name__, url_prefix="/api/spotify")
spotify_auth_bp = Blueprint("spotify_auth", __name__)

_spotify_app_token = None
_spotify_app_expires_at = 0


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
        return _spotify_app_token

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None

    auth = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    encoded = base64.b64encode(auth.encode("utf-8")).decode("utf-8")
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {encoded}"},
        timeout=5,
    )
    r.raise_for_status()
    data = r.json()
    _spotify_app_token = data.get("access_token")
    expires_in = data.get("expires_in", 3600)
    _spotify_app_expires_at = time.time() + int(expires_in)
    return _spotify_app_token


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


@spotify_bp.route("/audio-features", methods=["POST"])
def get_audio_features():
    """Get average audio features for a playlist - Improved with logging"""
    
    def log(msg):
        """Force stdout flush for immediate logging"""
        print(msg, flush=True)
        sys.stdout.flush()
    
    try:
        data = request.get_json()
        track_ids = data.get("track_ids", [])
        
        log("═══════════════════════════════════════")
        log("🎨 AUDIO FEATURES API - NEW REQUEST")
        log("═══════════════════════════════════════")
        log(f"📥 Received {len(track_ids)} track IDs")
        log(f"🔑 IDs: {track_ids}")
        
        if not track_ids:
            log("⚠️ No track IDs - returning defaults")
            return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
        
        # Validate
        valid_ids = [tid for tid in track_ids if isinstance(tid, str) and len(tid) == 22 and tid.isalnum()]
        log(f"✅ Valid IDs: {len(valid_ids)}/{len(track_ids)}")
        
        if not valid_ids:
            log("❌ No valid IDs - returning defaults")
            return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
        
        # Get token
        access_token, error = get_valid_access_token()
        if error:
            log(f"⚠️ User token failed: {error}")
            access_token = get_app_access_token()
            log("✅ Using app token" if access_token else "❌ No token available")
        else:
            log("✅ Using user token")
        
        if not access_token:
            return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
        
        # Call Spotify
        ids_str = ",".join(valid_ids[:100])
        log(f"📡 Calling Spotify API...")
        
        r = requests.get(
            "https://api.spotify.com/v1/audio-features",
            params={"ids": ids_str},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        
        log(f"📡 Response: {r.status_code}")
        
        if r.status_code != 200:
            log(f"❌ API Error: {r.text[:100]}")
            return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
        
        features = r.json().get("audio_features", [])
        valid = [f for f in features if f is not None]
        log(f"✅ Got {len(valid)}/{len(features)} valid features")
        
        if not valid:
            log("⚠️ No valid features")
            return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
        
        # Calculate
        avg = {
            "energy": sum(f.get("energy", 0) for f in valid) / len(valid),
            "danceability": sum(f.get("danceability", 0) for f in valid) / len(valid),
            "valence": sum(f.get("valence", 0) for f in valid) / len(valid),
            "acousticness": sum(f.get("acousticness", 0) for f in valid) / len(valid),
            "tempo": sum(f.get("tempo", 120) for f in valid) / len(valid),
        }
        
        log("📊 RESULTS:")
        log(f"  Energy: {avg['energy']:.3f}")
        log(f"  Dance: {avg['danceability']:.3f}")
        log(f"  Valence: {avg['valence']:.3f}")
        log(f"  Acoustic: {avg['acousticness']:.3f}")
        log(f"  Tempo: {avg['tempo']:.1f}")
        log("═══════════════════════════════════════")
        
        return jsonify(avg), 200
        
    except Exception as e:
        log(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({"energy": 0.5, "danceability": 0.5, "valence": 0.5, "acousticness": 0.5, "tempo": 120}), 200
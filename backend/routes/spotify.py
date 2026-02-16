import base64
import os
import secrets
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


def build_mood_tags(features_list):
    if not features_list:
        return []

    totals = {
        "danceability": 0.0,
        "energy": 0.0,
        "valence": 0.0,
        "acousticness": 0.0,
        "instrumentalness": 0.0,
        "liveness": 0.0,
        "tempo": 0.0,
    }

    count = 0
    for f in features_list:
        if not f:
            continue
        for key in totals:
            if f.get(key) is not None:
                totals[key] += float(f.get(key))
        count += 1

    if count == 0:
        return []

    avg = {k: totals[k] / count for k in totals}

    tags = []
    if avg["valence"] >= 0.6:
        tags.append("bright")
    elif avg["valence"] <= 0.4:
        tags.append("moody")
    else:
        tags.append("balanced")

    if avg["energy"] >= 0.7 or avg["tempo"] >= 125:
        tags.append("upbeat")
    elif avg["energy"] <= 0.35 or avg["tempo"] <= 90:
        tags.append("calm")
    else:
        tags.append("steady")

    if avg["acousticness"] >= 0.6:
        tags.append("acoustic")
    elif avg["instrumentalness"] >= 0.5:
        tags.append("instrumental")
    elif avg["danceability"] >= 0.65:
        tags.append("groovy")
    elif avg["liveness"] >= 0.5:
        tags.append("live")
    else:
        tags.append("smooth")

    unique = []
    for tag in tags:
        if tag not in unique:
            unique.append(tag)

    extras = []
    if avg["danceability"] >= 0.65:
        extras.append("groovy")
    if avg["acousticness"] >= 0.6:
        extras.append("acoustic")
    if avg["instrumentalness"] >= 0.5:
        extras.append("instrumental")
    if avg["liveness"] >= 0.5:
        extras.append("live")
    if avg["energy"] >= 0.7:
        extras.append("energetic")
    if avg["valence"] <= 0.35:
        extras.append("dark")

    for tag in extras:
        if tag not in unique:
            unique.append(tag)
        if len(unique) >= 3:
            break

    return unique[:3]


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


@spotify_bp.route("/mood-tags")
def spotify_mood_tags():
    ids = request.args.get("ids", "").strip()
    if not ids:
        return jsonify(success=True, tags=[])

    access_token, error = get_valid_access_token()
    if error:
        access_token = get_app_access_token()
        if not access_token:
            return jsonify(success=False, message="spotify token unavailable"), 401

    try:
        r = requests.get(
            "https://api.spotify.com/v1/audio-features",
            params={"ids": ids},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=5,
        )
        r.raise_for_status()
        features = r.json().get("audio_features", [])
    except Exception:
        return jsonify(success=True, tags=[])

    tags = build_mood_tags(features)
    return jsonify(success=True, tags=tags)


@spotify_bp.route("/audio-features", methods=["POST"])
def get_audio_features():
    """Get average audio features for a playlist"""
    data = request.get_json()
    track_ids = data.get("track_ids", [])
    
    if not track_ids:
        return jsonify({"error": "No track IDs provided"}), 400
    
    # Try user token first, fallback to app token
    access_token, error = get_valid_access_token()
    if error:
        access_token = get_app_access_token()
        if not access_token:
            return jsonify({"error": "Spotify token unavailable"}), 401
    
    try:
        # Spotify API: Get Audio Features for Several Tracks
        ids_str = ",".join(track_ids[:100])  # Max 100 tracks
        r = requests.get(
            "https://api.spotify.com/v1/audio-features",
            params={"ids": ids_str},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=5,
        )
        r.raise_for_status()
        
        features_data = r.json()
        audio_features = features_data.get("audio_features", [])
        
        # Calculate averages
        valid_features = [f for f in audio_features if f is not None]
        
        if not valid_features:
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            })
        
        avg_features = {
            "energy": sum(f.get("energy", 0) for f in valid_features) / len(valid_features),
            "danceability": sum(f.get("danceability", 0) for f in valid_features) / len(valid_features),
            "valence": sum(f.get("valence", 0) for f in valid_features) / len(valid_features),
            "acousticness": sum(f.get("acousticness", 0) for f in valid_features) / len(valid_features),
            "tempo": sum(f.get("tempo", 120) for f in valid_features) / len(valid_features),
        }
        
        return jsonify(avg_features)
    
    except Exception as e:
        print(f"Error fetching audio features: {e}")
        return jsonify({"error": str(e)}), 500
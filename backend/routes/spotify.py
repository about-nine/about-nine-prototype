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
    """
    Get average audio features for a playlist
    
    Spotify API Reference:
    https://developer.spotify.com/documentation/web-api/reference/get-several-audio-features
    
    Request body:
    {
        "track_ids": ["trackId1", "trackId2", ...]
    }
    
    Response:
    {
        "energy": 0.0-1.0,
        "danceability": 0.0-1.0,
        "valence": 0.0-1.0,
        "acousticness": 0.0-1.0,
        "tempo": BPM
    }
    """
    try:
        data = request.get_json()
        track_ids = data.get("track_ids", [])
        
        print("═══════════════════════════════════════")
        print("🎨 AUDIO FEATURES API REQUEST")
        print("═══════════════════════════════════════")
        print(f"📥 Received {len(track_ids)} track IDs")
        
        # Validate track IDs
        if not track_ids:
            print("⚠️ No track IDs provided")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        
        # Validate track ID format (22 characters, alphanumeric)
        valid_ids = []
        for tid in track_ids:
            if isinstance(tid, str) and len(tid) == 22 and tid.isalnum():
                valid_ids.append(tid)
            else:
                print(f"⚠️ Invalid track ID format: {tid}")
        
        if not valid_ids:
            print("❌ No valid track IDs found")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        
        print(f"✅ Valid track IDs: {len(valid_ids)}")
        print(f"🔑 Sample IDs: {valid_ids[:3]}")
        
        # Try user token first, fallback to app token
        access_token, error = get_valid_access_token()
        if error:
            print(f"⚠️ User token failed: {error}, trying app token...")
            access_token = get_app_access_token()
            if not access_token:
                print("❌ No access token available")
                return jsonify({
                    "energy": 0.5,
                    "danceability": 0.5,
                    "valence": 0.5,
                    "acousticness": 0.5,
                    "tempo": 120
                }), 200
            print("✅ Using app token")
        else:
            print("✅ Using user token")
        
        # Spotify API: Get Audio Features for Several Tracks
        # Max 100 tracks per request
        ids_str = ",".join(valid_ids[:100])
        
        print(f"📡 Calling Spotify API...")
        print(f"🔗 Endpoint: GET https://api.spotify.com/v1/audio-features")
        print(f"📊 Query params: ids={ids_str[:100]}...")
        
        try:
            r = requests.get(
                "https://api.spotify.com/v1/audio-features",
                params={"ids": ids_str},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            
            print(f"📡 Spotify API response: {r.status_code}")
            
            if r.status_code != 200:
                error_body = r.text[:200]
                print(f"❌ Spotify API error: {r.status_code}")
                print(f"📄 Error body: {error_body}")
                return jsonify({
                    "energy": 0.5,
                    "danceability": 0.5,
                    "valence": 0.5,
                    "acousticness": 0.5,
                    "tempo": 120
                }), 200
            
            response_data = r.json()
            audio_features = response_data.get("audio_features", [])
            
            print(f"✅ Got response from Spotify")
            print(f"📊 Total features: {len(audio_features)}")
            
        except requests.exceptions.Timeout:
            print("❌ Spotify API timeout")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Spotify API request failed: {e}")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        
        # Calculate averages (filter out None values)
        valid_features = [f for f in audio_features if f is not None]
        print(f"📊 Valid features: {len(valid_features)}/{len(audio_features)}")
        
        if not valid_features:
            print("⚠️ No valid features found (all None)")
            print(f"📋 Sample response: {audio_features[:3]}")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        
        # Calculate average for each feature
        total_energy = 0.0
        total_danceability = 0.0
        total_valence = 0.0
        total_acousticness = 0.0
        total_tempo = 0.0
        count = 0
        
        for feature in valid_features:
            if feature.get("energy") is not None:
                total_energy += float(feature.get("energy", 0))
            if feature.get("danceability") is not None:
                total_danceability += float(feature.get("danceability", 0))
            if feature.get("valence") is not None:
                total_valence += float(feature.get("valence", 0))
            if feature.get("acousticness") is not None:
                total_acousticness += float(feature.get("acousticness", 0))
            if feature.get("tempo") is not None:
                total_tempo += float(feature.get("tempo", 120))
            count += 1
        
        if count == 0:
            print("⚠️ No features to average")
            return jsonify({
                "energy": 0.5,
                "danceability": 0.5,
                "valence": 0.5,
                "acousticness": 0.5,
                "tempo": 120
            }), 200
        
        avg_features = {
            "energy": total_energy / count,
            "danceability": total_danceability / count,
            "valence": total_valence / count,
            "acousticness": total_acousticness / count,
            "tempo": total_tempo / count,
        }
        
        print("───────────────────────────────────────")
        print("📈 CALCULATED AVERAGES:")
        print(f"  Energy:       {avg_features['energy']:.3f}")
        print(f"  Danceability: {avg_features['danceability']:.3f}")
        print(f"  Valence:      {avg_features['valence']:.3f}")
        print(f"  Acousticness: {avg_features['acousticness']:.3f}")
        print(f"  Tempo:        {avg_features['tempo']:.1f} BPM")
        print("───────────────────────────────────────")
        
        # Check if we got default values (would indicate all features were 0.5)
        is_default = (
            abs(avg_features['energy'] - 0.5) < 0.01 and
            abs(avg_features['danceability'] - 0.5) < 0.01 and
            abs(avg_features['valence'] - 0.5) < 0.01 and
            abs(avg_features['acousticness'] - 0.5) < 0.01 and
            abs(avg_features['tempo'] - 120) < 1
        )
        
        if is_default:
            print("⚠️ WARNING: Results are suspiciously close to defaults")
            print("⚠️ This might indicate an issue with the source data")
        else:
            print("✅ Features calculated successfully")
        
        print("═══════════════════════════════════════")
        
        return jsonify(avg_features), 200
    
    except Exception as e:
        print(f"❌ Error in get_audio_features: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "energy": 0.5,
            "danceability": 0.5,
            "valence": 0.5,
            "acousticness": 0.5,
            "tempo": 120
        }), 200
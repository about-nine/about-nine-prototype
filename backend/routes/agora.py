# routes/agora.py - Agora recording control endpoints
from flask import Blueprint, request, jsonify
from backend.services.agora_service import (
    acquire_recording,
    start_recording,
    stop_recording,
    query_recording
)

from firebase_admin import db
import time

agora_bp = Blueprint("agora", __name__, url_prefix="/api/agora")


@agora_bp.route("/auto-start-recording", methods=["POST"])
def auto_start_recording():
    """양쪽이 ready되면 자동으로 녹화 시작"""
    try:
        data = request.get_json()
        request_id = data.get("request_id")
        
        if not request_id:
            return jsonify(success=False, message="request_id required"), 400
        
        ref = db.reference(f"match_requests/{request_id}")
        match_request = ref.get()
        
        if not match_request:
            return jsonify(success=False, message="Match request not found"), 404
        
        # 🔥 이미 녹화 중인지 확인 (중복 방지)
        if match_request.get("recording_sid"):
            print(f"⚠️ 이미 녹화 시작됨: {request_id}")
            return jsonify(success=True, message="Recording already started"), 200
        
        initiator_ready = match_request.get("initiator_ready", False)
        receiver_ready = match_request.get("receiver_ready", False)
        
        if not (initiator_ready and receiver_ready):
            return jsonify(success=False, message="Both users not ready"), 400
        
        print(f"🎬 자동 녹화 시작: {request_id}")
        
        # Acquire
        acquire_result = acquire_recording(request_id)
        
        if not acquire_result.get("success"):
            return jsonify(acquire_result), 500
        
        resource_id = acquire_result.get("resourceId")
        print(f"✅ resourceId: {resource_id}")
        
        # Start
        start_result = start_recording(request_id, resource_id)
        
        if not start_result.get("success"):
            return jsonify(start_result), 500
        
        sid = start_result.get("sid")
        print(f"✅ sid: {sid}")
        
        # Firebase 업데이트
        ref.update({
            "recording_resource": resource_id,
            "recording_sid": sid,
            "recording_started_at": int(time.time() * 1000),
            "call_started_at": int(time.time() * 1000)
        })
        
        print(f"✅ 녹화 시작 완료 및 Firebase 업데이트")
        
        return jsonify({
            "success": True,
            "resourceId": resource_id,
            "sid": sid
        })
    
    except Exception as e:
        print(f"❌ 자동 녹화 시작 실패: {e}")
        return jsonify(success=False, message=str(e)), 500


@agora_bp.route("/auto-stop-recording", methods=["POST"])
def auto_stop_recording():
    """통화 종료 시 자동으로 녹화 종료 (중복 방지 강화)"""
    try:
        data = request.get_json()
        request_id = data.get("request_id")
        
        if not request_id:
            return jsonify(success=False, message="request_id required"), 400
        
        ref = db.reference(f"match_requests/{request_id}")
        match_request = ref.get()
        
        if not match_request:
            return jsonify(success=False, message="Match request not found"), 404
        
        # 🔥 이미 녹화 종료되었는지 확인
        if match_request.get("recording_stopped"):
            print(f"⚠️ 이미 녹화 종료됨: {request_id}")
            return jsonify(success=True, message="Recording already stopped"), 200
        
        resource_id = match_request.get("recording_resource")
        sid = match_request.get("recording_sid")
        
        if not resource_id or not sid:
            print(f"⚠️ 녹화 정보 없음: resourceId={resource_id}, sid={sid}")
            # 🔥 녹화 없었으므로 stopped 플래그 설정
            ref.update({"recording_stopped": True})
            return jsonify(success=True, message="No recording to stop"), 200
        
        print(f"🛑 자동 녹화 종료: {request_id}")
        print(f"   resourceId: {resource_id}")
        print(f"   sid: {sid}")
        
        # 🔥 먼저 stopped 플래그 설정 (중복 방지)
        ref.update({"recording_stopped": True})
        
        # 녹화 종료
        stop_result = stop_recording(request_id, resource_id, sid)
        
        if not stop_result.get("success"):
            # 실패해도 플래그는 유지 (중복 호출 방지)
            return jsonify(stop_result), 500
        
        # 🔥 uploadingStatus 로깅
        uploading_status = stop_result.get("uploadingStatus", "unknown")
        print(f"📦 uploadingStatus: {uploading_status}")
        
        if uploading_status == "backuped":
            print(f"⚠️ Firebase Storage 업로드 실패! Agora 백업에만 저장됨")
        elif uploading_status == "uploaded":
            print(f"✅ Firebase Storage 업로드 성공!")
        
        # Firebase 업데이트
        ref.update({
            "recording_ended_at": int(time.time() * 1000),
            "recording_file_list": stop_result.get("fileList", []),
            "recording_uploading_status": uploading_status  # 🔥 추가
        })
        
        print(f"✅ 녹화 종료 완료")
        
        return jsonify({
            "success": True,
            "fileList": stop_result.get("fileList", []),
            "totalFiles": stop_result.get("totalFiles", 0),
            "uploadingStatus": uploading_status  # 🔥 추가
        })
    
    except Exception as e:
        print(f"❌ 자동 녹화 종료 실패: {e}")
        return jsonify(success=False, message=str(e)), 500


# =========================
# 수동 API (디버깅용)
# =========================
@agora_bp.route("/acquire", methods=["POST"])
def acquire():
    """수동 Acquire (디버깅용)"""
    data = request.get_json()
    channel = data.get("channel")
    
    if not channel:
        return jsonify(success=False, message="channel required"), 400
    
    result = acquire_recording(channel)
    return jsonify(result)


@agora_bp.route("/start", methods=["POST"])
def start():
    """수동 Start (디버깅용)"""
    data = request.get_json()
    channel = data.get("channel")
    resource_id = data.get("resourceId")
    
    if not channel or not resource_id:
        return jsonify(success=False, message="channel and resourceId required"), 400
    
    result = start_recording(channel, resource_id)
    return jsonify(result)


@agora_bp.route("/stop", methods=["POST"])
def stop():
    """수동 Stop (디버깅용)"""
    data = request.get_json()
    channel = data.get("channel")
    resource_id = data.get("resourceId")
    sid = data.get("sid")
    
    if not channel or not resource_id or not sid:
        return jsonify(success=False, message="channel, resourceId, and sid required"), 400
    
    result = stop_recording(channel, resource_id, sid)
    return jsonify(result)


@agora_bp.route("/query", methods=["POST"])
def query():
    """녹화 상태 조회"""
    data = request.get_json()
    resource_id = data.get("resourceId")
    sid = data.get("sid")
    
    if not resource_id or not sid:
        return jsonify(success=False, message="resourceId and sid required"), 400
    
    result = query_recording(resource_id, sid)
    return jsonify(result)

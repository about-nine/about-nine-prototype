# services/agora_service.py - Agora recording API integration
import json
import os
import base64
import requests

BUCKET = "about-nine-prototype-46a2c.firebasestorage.app"
GCS_ACCESS_KEY = os.getenv("GCS_ACCESS_KEY")
GCS_SECRET_KEY = os.getenv("GCS_SECRET_KEY")

if not GCS_ACCESS_KEY or not GCS_SECRET_KEY:
    raise RuntimeError("GCS_ACCESS_KEY and GCS_SECRET_KEY must be set!")

# Agora 설정
APP_ID = os.getenv("AGORA_APP_ID")
CUSTOMER_ID = os.getenv("AGORA_CUSTOMER_ID")
CUSTOMER_SECRET = os.getenv("AGORA_CUSTOMER_SECRET")

RECORDING_BOT_UID = "999"


def _auth_header():
    """Agora API Basic Auth 헤더"""
    token = base64.b64encode(
        f"{CUSTOMER_ID}:{CUSTOMER_SECRET}".encode()
    ).decode()
    
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


def acquire_recording(channel: str):
    """1단계: Resource ID 획득"""
    url = f"https://api.agora.io/v1/apps/{APP_ID}/cloud_recording/acquire"
    
    payload = {
        "cname": channel,
        "uid": RECORDING_BOT_UID,
        "clientRequest": {
            "resourceExpiredHour": 24,
            "scene": 0
        }
    }
    
    print(f"📤 Agora Acquire 요청: {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=_auth_header(), timeout=10)
        result = response.json()
        
        print(f"📥 Agora Acquire 응답: {response.status_code}, {result}")
        
        if response.status_code == 200:
            return {
                "success": True,
                "resourceId": result.get("resourceId"),
                "cname": result.get("cname"),
                "uid": result.get("uid")
            }
        else:
            return {
                "success": False,
                "message": result
            }
    
    except Exception as e:
        print(f"❌ Acquire 실패: {e}")
        return {
            "success": False,
            "message": str(e)
        }


def start_recording(channel: str, resource_id: str):
    """
    2단계: 녹화 시작 (Individual 모드 - 화자 분리)
    """
    url = (
        f"https://api.agora.io/v1/apps/{APP_ID}"
        f"/cloud_recording/resourceid/{resource_id}/mode/individual/start"
    )
    
    payload = {
        "cname": channel,
        "uid": RECORDING_BOT_UID,
        "clientRequest": {
            "token": "",
            "recordingConfig": {
                "channelType": 0,
                "streamTypes": 0,
                "maxIdleTime": 120,
                "streamMode": "original",
                "subscribeAudioUids": ["#allstream#"],
                "subscribeUidGroup": 0,
            },
            "storageConfig": {
                "vendor": 6,
                "region": 0,  # ✅ Google Cloud는 무조건 0
                "bucket": BUCKET,
                "accessKey": GCS_ACCESS_KEY,
                "secretKey": GCS_SECRET_KEY,
                "fileNamePrefix": ["recordings"]
            }
        }
    }
    
    print(f"📤 Agora Start (Individual) 요청")
    print(f"   Bucket: {BUCKET}")
    print(f"   Region: 0 (Google Cloud)")
    print(f"   Prefix: recordings")
    
    try:
        response = requests.post(url, json=payload, headers=_auth_header(), timeout=10)
        result = response.json()
        
        print(f"📥 Agora Start 응답: {response.status_code}, {result}")
        
        if response.status_code == 200:
            return {
                "success": True,
                "sid": result.get("sid"),
                "resourceId": result.get("resourceId")
            }
        else:
            return {
                "success": False,
                "message": result
            }
    
    except Exception as e:
        print(f"❌ Start 실패: {e}")
        return {
            "success": False,
            "message": str(e)
        }


def stop_recording(channel: str, resource_id: str, sid: str):
    """3단계: 녹화 종료 (Individual 모드)"""
    url = (
        f"https://api.agora.io/v1/apps/{APP_ID}"
        f"/cloud_recording/resourceid/{resource_id}/sid/{sid}/mode/individual/stop"
    )
    
    payload = {
        "cname": channel,
        "uid": RECORDING_BOT_UID,
        "clientRequest": {}
    }
    
    print(f"📤 Agora Stop (Individual) 요청: channel={channel}")
    
    try:
        response = requests.post(url, json=payload, headers=_auth_header(), timeout=10)
        result = response.json()
        
        print(f"📥 Agora Stop 응답: {response.status_code}")
        print(f"   Full response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            server_response = result.get("serverResponse", {})
            file_list_raw = server_response.get("fileList", [])
            file_list_mode = server_response.get("fileListMode", "json")
            uploading_status = server_response.get("uploadingStatus", "unknown")
            
            # 🔥 uploadingStatus 상세 로그
            print(f"📦 uploadingStatus: {uploading_status}")
            if uploading_status == "backuped":
                print(f"⚠️ Firebase Storage 업로드 실패!")
                print(f"   Bucket: {BUCKET}")
            elif uploading_status == "uploaded":
                print(f"✅ Firebase Storage 업로드 성공!")
            
            parsed_files = []
            for file in file_list_raw:
                file_name = file.get("fileName", "")
                uid = file.get("uid", "")
                
                parsed_file = {
                    "fileName": file_name,
                    "uid": uid,
                    "trackType": file.get("trackType", ""),
                    "mixedAllUser": file.get("mixedAllUser", False),
                    "isPlayable": file.get("isPlayable", True),
                    "sliceStartTime": file.get("sliceStartTime", 0),
                    "storagePath": file_name
                }
                parsed_files.append(parsed_file)
                
                print(f"📁 파일 {len(parsed_files)}:")
                print(f"   fileName: {parsed_file['fileName']}")
                print(f"   uid: {parsed_file['uid']}")
                print(f"   storagePath: {parsed_file['storagePath']}")
            
            print(f"✅ 총 {len(parsed_files)}개 파일 생성됨 (화자 분리)")
            
            return {
                "success": True,
                "fileList": parsed_files,
                "uploadingStatus": uploading_status,
                "fileListMode": file_list_mode,
                "totalFiles": len(parsed_files)
            }
        else:
            print(f"❌ Stop 실패: {result}")
            return {
                "success": False,
                "message": result
            }
    
    except Exception as e:
        print(f"❌ Stop 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e)
        }


def query_recording(resource_id: str, sid: str):
    """녹화 상태 조회 (Individual 모드)"""
    url = (
        f"https://api.agora.io/v1/apps/{APP_ID}"
        f"/cloud_recording/resourceid/{resource_id}/sid/{sid}/mode/individual/query"
    )
    
    try:
        response = requests.get(url, headers=_auth_header(), timeout=10)
        result = response.json()
        
        print(f"📥 Agora Query 응답: {response.status_code}, {result}")
        
        return result
    
    except Exception as e:
        print(f"❌ Query 실패: {e}")
        return {
            "success": False,
            "message": str(e)
        }
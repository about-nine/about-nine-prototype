import { rtdb, ref, set, onDisconnect, onValue } from "/js/firebase.js";

/* =========================
   Presence Init
========================= */

let presenceInitialized = false;
let presenceRetryTimer = null;

function resolveUserId() {
  let userId = null;
  try {
    const raw = localStorage.getItem("user_id");
    if (raw) {
      try {
        userId = JSON.parse(raw);
      } catch {
        userId = raw;
      }
    }
  } catch {}
  return userId;
}

function initPresence() {
  if (presenceInitialized) return;
  const userId = resolveUserId();
  if (!userId) return;
  presenceInitialized = true;
  if (presenceRetryTimer) clearInterval(presenceRetryTimer);
  presenceRetryTimer = null;

  const presenceRef = ref(rtdb, "presence/" + userId);
  const connectedRef = ref(rtdb, ".info/connected");
  let heartbeatTimer = null;

  const setOnline = () =>
    set(presenceRef, { online: true, updated_at: Date.now() }).catch((e) => {
      console.error("presence setOnline failed:", e);
    });
  const setOffline = () =>
    set(presenceRef, { online: false, updated_at: Date.now() }).catch((e) => {
      console.error("presence setOffline failed:", e);
    });

  const startHeartbeat = () => {
    setOnline();
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = setInterval(() => {
      setOnline();
    }, 15000);
  };

  const stopHeartbeat = () => {
    if (heartbeatTimer) clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  };

  startHeartbeat();

  onValue(connectedRef, (snap) => {
    if (!snap.val()) return;
    onDisconnect(presenceRef).set({
      online: false,
      updated_at: Date.now(),
    });
    startHeartbeat();
    console.log("presence connected");
  });

  // 탭이 숨겨지면 heartbeat만 멈춤 (TTL로 자연스럽게 offline 판정)
  // 탭이 다시 보이면 heartbeat 재시작
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopHeartbeat();
    } else {
      startHeartbeat();
    }
  });

  // blur/focus는 제거하거나, focus에서만 heartbeat 재시작
  window.addEventListener("focus", startHeartbeat);
  // blur에서 setOffline() 호출하지 않음

  window.addEventListener("pageshow", startHeartbeat);
  window.addEventListener("pagehide", () => {
    try {
      setOffline();
    } catch {}
  });

  window.addEventListener("beforeunload", () => {
    try {
      setOffline();
    } catch {}
  });
}

/* =========================
   Run
========================= */

initPresence();
if (!presenceInitialized) {
  presenceRetryTimer = setInterval(() => {
    if (!presenceInitialized) initPresence();
  }, 1000);
}

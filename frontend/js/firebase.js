/* =========================
   Firebase Core
========================= */

import { initializeApp } 
  from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";

/* =========================
   Auth
========================= */

import {
  getAuth,
  RecaptchaVerifier,
  signInWithPhoneNumber,
  PhoneAuthProvider,
  signInWithCredential
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

/* =========================
   Realtime DB
========================= */

import {
  getDatabase,
  ref,
  set,
  get,
  update,
  push,
  remove,
  onDisconnect,
  onValue
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-database.js";

/* =========================
   Firestore (separate sdk)
========================= */

import { getFirestore }
  from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";


/* =========================
   Config
========================= */

const firebaseConfig = {
  apiKey: "AIzaSyDC0zepbZP9uOd8uHPLMr3-1nKjaelwtu4",
  authDomain: "about-nine-prototype-46a2c.firebaseapp.com",
  databaseURL:
    "https://about-nine-prototype-46a2c-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "about-nine-prototype-46a2c",
  storageBucket: "about-nine-prototype-46a2c.firebasestorage.app",
  messagingSenderId: "746184089802",
  appId: "1:746184089802:web:741d74b2cb1f5fe4231433"
};


/* =========================
   Initialize (ONLY ONCE)
========================= */

const app = initializeApp(firebaseConfig);

const TEMP_STORAGE = {};

/* =========================
   Export Instances
========================= */

export const auth = getAuth(app);
export const db = getFirestore(app);
export const rtdb = getDatabase(app);

// 🔥 Realtime Database 함수들 export
export { ref, set, get, update, push, remove, onDisconnect, onValue };


/* =====================================================
   Phone Auth Helpers
===================================================== */

/* ---------- Recaptcha ---------- */

window.initRecaptcha = () => {
  window.recaptchaVerifier = new RecaptchaVerifier(
    auth,
    "recaptcha-container",
    { size: "invisible" }
  );
};


/* ---------- Send OTP ---------- */

window.sendFirebaseOTP = async (phone) => {
  const confirmation = await signInWithPhoneNumber(
    auth,
    phone,
    window.recaptchaVerifier
  );

  try {
    localStorage.setItem("verificationId", confirmation.verificationId);
  } catch {
    TEMP_STORAGE.verificationId = confirmation.verificationId;
  }
};


/* ---------- Verify OTP ---------- */

window.verifyFirebaseOTP = async (code) => {
  let verificationId = null;
  try {
    verificationId = localStorage.getItem("verificationId");
  } catch {
    verificationId = TEMP_STORAGE.verificationId || null;
  }

  const credential =
    PhoneAuthProvider.credential(verificationId, code);

  const result =
    await signInWithCredential(auth, credential);

  return await result.user.getIdToken();
};

window.getFirebaseIdToken = async (forceRefresh = false) => {
  const user = auth.currentUser;
  if (!user) return null;
  try {
    return await user.getIdToken(forceRefresh);
  } catch {
    return null;
  }
};

import { signOut as firebaseSignOut } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

window.firebaseLogout = async () => {
  try {
    await firebaseSignOut(auth);
  } catch (e) {
    console.error("Firebase signOut error:", e);
  }
};

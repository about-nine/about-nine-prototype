const MEMORY_STORE = {};

function getLocalStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function rawSetItem(key, value) {
  const store = getLocalStorage();
  if (store) {
    try {
      store.setItem(key, value);
      return;
    } catch {}
  }
  MEMORY_STORE[key] = value;
}

function rawGetItem(key) {
  const store = getLocalStorage();
  if (store) {
    try {
      return store.getItem(key);
    } catch {}
  }
  return Object.prototype.hasOwnProperty.call(MEMORY_STORE, key)
    ? MEMORY_STORE[key]
    : null;
}

function rawRemoveItem(key) {
  const store = getLocalStorage();
  if (store) {
    try {
      store.removeItem(key);
      return;
    } catch {}
  }
  delete MEMORY_STORE[key];
}

const API_BASE = (() => {
  const params = new URLSearchParams(window.location.search);
  const paramBase = params.get("apiBase");
  if (paramBase) {
    rawSetItem("api_base", paramBase);
  }
  const stored = rawGetItem("api_base");
  if (stored) return stored;
  if (window.__API_BASE__) return window.__API_BASE__;
  return `${window.location.origin}/api`;
})();

// Ensure manifest + icons exist so every static page is PWA-ready.
(function ensurePWAAssets() {
  const head = document.head || document.getElementsByTagName("head")[0];
  if (!head) return;

  const upsertLink = (attrs) => {
    if (head.querySelector(`[rel="${attrs.rel}"][href="${attrs.href}"]`)) return;
    const link = document.createElement("link");
    Object.entries(attrs).forEach(([key, value]) => (link[key] = value));
    head.appendChild(link);
  };

  const ensureMeta = (name, content) => {
    let meta = head.querySelector(`meta[name="${name}"]`);
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = name;
      head.appendChild(meta);
    }
    meta.content = content;
  };

  upsertLink({ rel: "manifest", href: "/manifest.json" });
  upsertLink({ rel: "icon", href: "/images/icon-192.png", sizes: "192x192", type: "image/png" });
  upsertLink({ rel: "icon", href: "/images/icon-512.png", sizes: "512x512", type: "image/png" });
  upsertLink({
    rel: "apple-touch-icon",
    href: "/images/icon-192.png",
    sizes: "192x192",
    type: "image/png",
  });
  ensureMeta("theme-color", "#000000");
})();

// API 호출 헬퍼
async function getAuthToken() {
  if (typeof window.getFirebaseIdToken === "function") {
    try {
      const token = await window.getFirebaseIdToken();
      if (token) {
        saveToLocal("id_token", token);
        return token;
      }
    } catch {}
  }
  const cached = getFromLocal("id_token");
  return cached || null;
}

async function apiCall(endpoint, method = "GET", data = null) {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
    credentials: "include",
  };

  const userId = getFromLocal("user_id");
  if (userId) options.headers["X-User-ID"] = userId;

  const idToken = await getAuthToken();
  if (idToken) options.headers["Authorization"] = `Bearer ${idToken}`;

  if (data && method !== "GET") options.body = JSON.stringify(data);

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      const text = await response.text().catch(() => "");
      const err = new Error(
        text || `unexpected response (${response.status})`,
      );
      err.status = response.status;
      err.body = text;
      throw err;
    }

    const result = await response.json().catch(() => null);
    if (!response.ok) {
      const err = new Error(result?.message || "요청 실패");
      err.status = response.status;
      err.data = result;
      throw err;
    }
    if (result === null) {
      const err = new Error("invalid server response");
      err.status = response.status;
      throw err;
    }
    return result;
  } catch (error) {
    console.error("API 호출 오류:", error);
    throw error;
  }
}

// 로컬 스토리지 헬퍼
function saveToLocal(key, value) {
  try {
    rawSetItem(key, JSON.stringify(value));
  } catch (error) {
    console.error("localStorage 저장 오류:", error);
  }
}

function getFromLocal(key) {
  let v;
  try {
    v = rawGetItem(key);
  } catch {
    v = null;
  }
  if (!v) return null;
  try { return JSON.parse(v); } catch { return v; }
}

function removeFromLocal(key) {
  try {
    rawRemoveItem(key);
  } catch (error) {
    console.error("localStorage 삭제 오류:", error);
  }
}

// 페이지 이동
function navigateTo(page) {
  window.location.href = page;
}

// 로딩 표시
function showLoading(text = "Loading...") {
  let overlay = document.getElementById("loading-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loading-overlay";
    overlay.innerHTML = `
      <div class="loading-asterisks">
        <img src="images/asterisk.png" class="asterisk asterisk-1" alt="loading">
        <img src="images/asterisk.png" class="asterisk asterisk-2" alt="loading">
      </div>
      <div class="loading-text">${text}</div>
    `;
    document.body.appendChild(overlay);
  } else {
    overlay.querySelector(".loading-text").textContent = text;
    overlay.classList.remove("hidden");
  }
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("hidden");
}

// 에러 메시지 표시
function showError(elementId, message) {
  const errorDiv = document.getElementById(elementId);
  if (errorDiv) {
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
  }
}

function hideError(elementId) {
  const errorDiv = document.getElementById(elementId);
  if (errorDiv) {
    errorDiv.textContent = "";
    errorDiv.classList.add("hidden");
  }
}

// 유틸리티
function formatPhoneNumber(value) {
  const numbers = value.replace(/[^0-9]/g, "");
  if (numbers.length <= 3) return numbers;
  if (numbers.length <= 7) return numbers.slice(0, 3) + "-" + numbers.slice(3);
  return numbers.slice(0, 3) + "-" + numbers.slice(3, 7) + "-" + numbers.slice(7, 11);
}

function calculateAge(birthDate) {
  const [y, m, d] = birthDate.split("-").map(Number);
  const birth = new Date(y, m - 1, d);
  const today = new Date();
  const age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    return age - 1;
  }
  return age;
}

/**
 * Promise 기반 sleep
 * talk-result.html, talk-end.html, option-select.html에서 중복 정의되던 것을 통합
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 순서 표현 (1st, 2nd, 3rd, ...)
 * history-detail.html, talk-result.html에서 중복 정의되던 것을 통합
 */
function getOrdinal(n) {
  if (n === 1) return "1st";
  if (n === 2) return "2nd";
  if (n === 3) return "3rd";
  return `${n}th`;
}

/**
 * 라운드 토픽 레이블
 * history.html, history-detail.html, talk-result.html에서 중복 정의되던 것을 통합
 */
function roundLabel(topic) {
  if (topic === "life") return "life";
  return topic || "";
}

/**
 * 대화 가이드 질문 목록
 * option-select.html, voice-talk.html에서 동일한 객체가 통째로 중복되던 것을 통합
 */
const GUIDE_QUESTIONS = {
  life: [
    "What's more important in a long-term relationship: stability or excitement?",
    "What's your biggest non-negotiable value in life?",
    "Do you admire your parents' relationship or want something completely different?",
    "Do you believe in second chances after cheating?",
    'What does the phrase "living your best life" mean to you?',
    "Do you think marriage is essential in life?",
    "When your values conflict with your parents' values, whose do you follow?",
    "What's your top priority in life: money, love, or success?",
    "Can you give up reality for your dreams?",
    "What's the proudest moment of your life?",
    "If you had to choose between protecting your family or your partner, which would it be?",
    "Would you rather have the same career for life, or completely change careers every 10 years?",
    'How do you define a "happy life"?',
    "What's the scariest thing about life to you?",
    "If someone loves you intensely but could make you unhappy, would you still choose them?",
    "Do you have any secrets you're keeping from your family?",
    "Would you give up love for success?",
    'Do you believe in "destined encounters"?',
    "If you could achieve only one thing in your entire life, what would you choose?",
    "Do you think people can truly change?",
    "When in your life did you feel most alive, and why?",
    "If you could erase one regret, would you do it—or do regrets shape who you are?",
    "Do you think happiness comes more from achieving goals or from appreciating what you already have?",
    "What's a value you would never compromise on, even for someone you love deeply?",
  ],
  travel: [
    "Would you rather road-trip across the U.S. or backpack through Asia?",
    "What's more fun: theme parks or national parks?",
    "How do you feel about camping—romantic or miserable?",
    "Do you prefer tourist spots or hidden gems?",
    "Would you rather do skydiving on vacation or a couples' spa?",
    "Which do you prefer: beach or mountains?",
    "Urban travel or nature travel—which appeals to you more?",
    "Do you like spontaneous trips or thorough planning?",
    "For accommodations, luxury hotel or guesthouse?",
    "If you fight with your partner during a trip, how do you resolve it?",
    "Do you want to do adventurous activities when traveling?",
    "What would you do if your partner has a completely different travel style?",
    "Trying local food vs. sticking to familiar food—which one?",
    "If you could only travel to one country for the rest of your life, where would you go?",
    "Do you think traveling with someone reveals their true personality more than daily life does?",
    "What's one place that changed the way you see the world?",
    "Would you rather visit every country once, or return to one place you love every year?",
    "Do you believe where you go matters more, or who you go with?",
    "If you could design the perfect couples' trip, what emotions would you want it to create—romance, thrill, peace, or discovery?",
  ],
  money: [
    "Do you think expensive gifts show love, or is thoughtfulness more important?",
    "How would you feel if your partner made way more money than you?",
    "Would you rather invest in a home or travel the world?",
    "If you won the lottery, what's the first thing you'd spend on with your partner?",
    "Do you think money can buy happiness?",
    "Do you think love can be shaken because of money?",
    "If it's something you want to do for life but pays nothing, would you still do it?",
    "If you had to choose between money or time, which would it be?",
    "How much do you think you should know about your partner's financial situation?",
    "When traveling, what's more important: value for money or luxury?",
    "When do you think is the best moment to spend money?",
    "Do you believe you can have a joyful relationship even if you're poor?",
    "If you have more money, does love grow deeper too?",
    "Do you think you can prove love with money?",
    "If you and your partner earned very different salaries, how would you want to handle it?",
    "What's one financial decision you'd want to make together as a couple, not alone?",
    "Would you rather be with someone generous but irresponsible, or disciplined but stingy?",
  ],
  time: [
    "Is daily texting a must, or are you fine with space?",
    "How would you spend a perfect Sunday together?",
    "Do you think long-distance can work if you're both busy?",
    "Are birthdays and anniversaries a big deal to you, or not really?",
    "Would you rather binge a Netflix series together or go out every night?",
    "What feels longer—waiting for a text back or sitting in traffic?",
    "Is being on time important to you, or are you usually casual about it?",
    "Do you prefer spontaneous dates or scheduled ones?",
    "Would you give up sleep for more time with your partner?",
    "Are you a morning person or a night owl?",
    "Which do you prefer: morning walks or late-night drives?",
    "Do you spend weekends actively or just relaxing?",
    "Can you tolerate a partner who's bad with time management?",
    "Would you wait for a partner who shows up an hour late to a date?",
    "What time of day do you most want to spend with your partner?",
    'If your partner says "I\'m too busy with work" on a special occasion, can you understand?',
    "Do you need alone time, or do you prefer being together constantly?",
    "How do you feel about spending all day with your partner?",
    "Do you think how someone spends their free time reveals who they truly are?",
    "What's a moment in time you wish you could pause forever?",
    "Do you see time as something you own, or something that owns you?",
    'If love had an "expiration date," how long would be enough for you to feel it was worth it?',
    "Do you think people grow apart more because of lost time together or because of changed priorities?",
  ],
  love: [
    "Do you believe in love at first sight?",
    "Do you like grand romantic gestures or subtle daily ones?",
    "How important is kissing compatibility?",
    'Do you think it\'s normal to say "I love you" within a few months?',
    "Do you think sexual chemistry can make or break a relationship?",
    "What's sexier: confidence or mystery?",
    "Would you ever mix romance with a little bit of risk (like in public)?",
    "Are you okay with public displays of affection?",
    "Do you prefer expressing affection with words or actions?",
    "Are you more proactive when it comes to physical affection?",
    "Do you feel hurt if your partner declines physical affection when you want it?",
    "In physical affection, are you the one who leads or prefers to be led?",
    "If you could only have one form of affection for life, what would you choose? (kissing, hugging, holding hands, etc.)",
    "Do you think love is a choice we make every day, or a feeling we can't control?",
    "What's the smallest gesture from a partner that makes you feel deeply loved?",
    "Do you think long-lasting passion is real, or does it inevitably fade?",
    "Would you rather have a partner who excites you endlessly but is unstable, or one who is steady but less thrilling?",
    "What scares you more in love—being vulnerable or being taken for granted?",
  ],
  relationship: [
    "Do you believe arguments make a relationship stronger or weaker?",
    "What's your biggest dealbreaker in a relationship?",
    "Do you think jealousy is a sign of love or insecurity?",
    "Should couples share passwords or keep digital privacy?",
    'Do you believe in "breaks" in relationships, or is that just a breakup in disguise?',
    "How do you usually handle conflict: talk it out, cool off, or avoid it?",
    "How much independence should partners have in a relationship?",
    'Do you believe "opposites attract," or do similar personalities last longer?',
    "What makes you feel most secure in a relationship—words, actions, or consistency?",
    "Do you think long-term relationships should feel easy, or is effort always required?",
    "Do you think love is proven more by how someone treats you on good days or on bad days?",
    "Do you think trust, once broken, can ever be fully restored?",
    "Would you rather have a partner who challenges you constantly or one who always supports you?",
    "Do you think the strongest relationships are built on similarity, or on learning to navigate differences?",
  ],
};

// 페이지 로드 시 초기화
document.addEventListener("DOMContentLoaded", () => {
  console.log("Common.js 로드 완료");
});

/**
 * 하단 네비게이션 렌더링
 * @param {string} activePage - 'lounge' | 'history' | 'profile'
 */
function renderBottomNav(activePage = "lounge") {
  const nav = document.getElementById("bottomNav");
  if (!nav) return;

  nav.innerHTML = `
    <button class="nav-item ${activePage === "lounge" ? "active" : ""}" onclick="navigateTo('lounge.html')">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    </button>
    <button class="nav-item ${activePage === "history" ? "active" : ""}" onclick="navigateTo('history.html')">
      <img src="images/asterisk.png" alt="asterisk" style="width:24px;height:24px;${activePage === "history" ? "filter:brightness(1.2);" : "filter:brightness(0.6);"}">
    </button>
    <button class="nav-item ${activePage === "profile" ? "active" : ""}" onclick="navigateTo('edit-profile.html')">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </button>
  `;
}

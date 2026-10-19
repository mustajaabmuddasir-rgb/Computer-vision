from __future__ import annotations

import os
import re
import time
import random
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ==============================================================================
# 1. CONSTANTS & MODEL CONFIGURATION
# ==============================================================================
PHONE_CLASS_ID = 67   # COCO: cell phone
BOOK_CLASS_ID = 73    # COCO: book
PERSON_CLASS_ID = 0   # COCO: person

DEFAULT_CHECK_INTERVAL = 1.0
DEFAULT_WARN_AFTER_SECONDS = 5
DEFAULT_ALERT_AFTER_SECONDS = 20
DEFAULT_FORGIVENESS_SECONDS = 4
DEFAULT_ALERT_COOLDOWN = 10 * 60

# Model weight resolution
DESKTOP_WEIGHTS_DIR = Path(r"C:\Users\Mustajaab\Desktop\YOLO Object Detection")


def resolve_model_path(family: str, variant: str) -> str:
    """
    YOLOv6 model configuration.

    Default model:
        YOLOv6n -> yolov6n.pt

    The function first checks the Desktop weights directory.
    If the file is not there, the filename is returned.
    """
    prefix = "yolov6"
    filename = f"{prefix}{variant}.pt"

    desktop_candidate = DESKTOP_WEIGHTS_DIR / filename

    if desktop_candidate.exists():
        return str(desktop_candidate)

    return filename


# ==============================================================================
# 2. NATURAL COMMENTARY REPERTOIRE
# ==============================================================================
COMMENTS_GOOD = [
    "Look at you diving into that book! Readers are future leaders! 📚✨",
    "Brilliant focus, {name}! Your brain is expanding with every page! 🧠🔥",
    "Outstanding work! That knowledge is going to take you far! 🌟",
    "Books over screens any day! Super proud of your dedication right now! 📖🏆",
    "Deep focus zone unlocked! Keep up the marvelous reading flow! 🚀",
    "Great study posture and focus! You're making learning look effortless! 💎",
    "Chapter champion! Every minute invested in books pays huge dividends! 📚🎯",
]

COMMENTS_BAD = [
    "Hey {name}! Put the phone away, let's keep our eyes on the prize! 📱👀",
    "Distraction spotted! The phone won't help you ace tomorrow's goals! ⚠️",
    "Screen time alert! Put it face down and let's get back to studying! 🛑",
    "Caught red-handed with the phone! Finish this study sprint first! 📱",
    "Don't let the notifications win, {name}! You're stronger than the scroll! 📵💪",
    "Whoops! Looks like a phone slipped into the study zone. Quick, put it away! ⏳",
]

COMMENTS_ALERT = [
    "🚨 Distraction threshold reached! Alert notification dispatched to parent.",
    "⚠️ Consecutive phone alerts logged! Time for a quick break or serious refocus.",
    "📢 Guardian alert fired! Let's swap the phone for a book immediately.",
]

COMMENTS_NEUTRAL = [
    "Desk is calm and ready. Let's pick up that book and crush today! 🌿",
    "Guardian is keeping watch. Take a deep breath and start when you're ready! ✨",
    "Study space is peaceful. Looking forward to some great reading! 📖",
    "Standing by! Show me what you're learning today, {name}! 🎓",
]


# ==============================================================================
# 3. HELPER FUNCTIONS & SMS INTEGRATION
# ==============================================================================
def normalize_pk_mobile(value: str) -> str | None:
    """Accept Pakistani mobile numbers or international E.164 numbers."""
    if not value:
        return None

    number = re.sub(r"[\s()\-]", "", value)

    if re.fullmatch(r"03\d{9}", number):
        return "+92" + number[1:]

    if re.fullmatch(r"\+923\d{9}", number):
        return number

    if re.fullmatch(r"\+\d{10,15}", number):
        return number

    return None


def send_sms(to_number: str, body: str) -> str:
    sid = os.getenv("TWILIO_ACCOUNT_SID") or st.session_state.get("twilio_sid", "")
    token = os.getenv("TWILIO_AUTH_TOKEN") or st.session_state.get("twilio_token", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER") or st.session_state.get("twilio_from", "")

    if not all((sid, token, from_number)):
        return "Not sent: Twilio credentials are not fully configured in Settings."

    try:
        from twilio.rest import Client

        client = Client(sid, token)

        message = client.messages.create(
            to=to_number,
            from_=from_number,
            body=body,
        )

        return f"Sent successfully (Twilio ID: {message.sid})."

    except Exception as exc:
        return f"Twilio dispatch failed: {str(exc).splitlines()[0]}"


@st.cache_resource(show_spinner="Loading YOLOv6n model...")
def load_detector(model_path: str) -> tuple[YOLO, threading.Lock]:
    model = YOLO(model_path)
    lock = threading.Lock()
    return model, lock


# ==============================================================================
# 4. SESSION STATE ARCHITECTURE
# ==============================================================================
def init_state() -> None:
    defaults = {
        "camera": None,
        "camera_index": 0,
        "monitoring": False,
        "kid_name": "Champion",
        "book_streak": 0,
        "phone_episode_start": None,
        "phone_last_seen": None,
        "phone_warn_tier": 0,
        "warn_after_seconds": DEFAULT_WARN_AFTER_SECONDS,
        "alert_after_seconds": DEFAULT_ALERT_AFTER_SECONDS,
        "forgiveness_seconds": DEFAULT_FORGIVENESS_SECONDS,
        "check_interval": DEFAULT_CHECK_INTERVAL,
        "feed_refresh_seconds": 0.15,
        "cooldown_seconds": DEFAULT_ALERT_COOLDOWN,
        "phone_conf": 0.40,
        "book_conf": 0.35,

        # ==========================================================================
        # YOLOv6n
        # ==========================================================================
        "model_family": "v6",
        "model_variant": "n",

        "infer_size": 480,
        "episode_handled": False,
        "last_check_at": float("-inf"),
        "last_alert_at": float("-inf"),
        "sms_status": "No alerts dispatched yet",
        "parent_phone": "",
        "events": [],
        "total_phone_alerts": 0,
        "total_books_read": 0,
        "reading_start_time": None,
        "total_reading_seconds": 0.0,
        "congratulations_count": 0,
        "last_comment": {
            "text": "Desk is calm and ready. Let's pick up that book and crush today! 🌿",
            "type": "neutral",
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "badge": "Standby",
        },
        "comment_history": deque(maxlen=20),
        "sound_enabled": True,
        "speech_enabled": False,
        "simulation_mode": False,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def record_event(
    category: str,
    title: str,
    details: str,
    severity: str = "info",
) -> None:
    """Store a structured event in history."""

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "id": len(st.session_state.events) + 1,
        "timestamp": ts,
        "category": category,
        "title": title,
        "details": details,
        "severity": severity,
    }

    st.session_state.events.insert(0, record)


def set_comment(category: str, custom_text: str | None = None) -> None:
    """Update the live activity commentary panel."""

    name = st.session_state.get("kid_name", "Champion")
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

    if custom_text:
        text = custom_text
        badge = "Notice"

    elif category == "good":
        text = random.choice(COMMENTS_GOOD).format(name=name)
        badge = "Reading focus"

    elif category == "bad":
        text = random.choice(COMMENTS_BAD).format(name=name)
        badge = "Distraction"

    elif category == "alert":
        text = random.choice(COMMENTS_ALERT).format(name=name)
        badge = "Parent alert"

    else:
        text = random.choice(COMMENTS_NEUTRAL).format(name=name)
        badge = "Quiet study"

    st.session_state.last_comment = {
        "text": text,
        "type": category,
        "time": now_str,
        "badge": badge,
    }

    st.session_state.comment_history.appendleft(
        st.session_state.last_comment
    )


def close_camera() -> None:
    if st.session_state.camera is not None:
        try:
            st.session_state.camera.release()
        except Exception:
            pass

        st.session_state.camera = None


# ==============================================================================
# 5. VISION & DETECTION ENGINE
# ==============================================================================
def detect_objects(
    frame,
    phone_conf: float,
    book_conf: float,
    model_path: str,
    infer_size: int,
) -> tuple[bool, bool, bool, np.ndarray, list]:

    """
    Runs YOLO inference for cell phones, books, and persons.
    Returns:
        phone_detected,
        book_detected,
        person_detected,
        annotated_frame,
        detections_summary
    """

    model, model_lock = load_detector(model_path)

    with model_lock:
        results = model.predict(
            frame,
            classes=[
                PHONE_CLASS_ID,
                BOOK_CLASS_ID,
                PERSON_CLASS_ID,
            ],
            conf=min(phone_conf, book_conf, 0.30),
            imgsz=infer_size,
            verbose=False,
        )[0]

    annotated = frame.copy()

    phone_found = False
    book_found = False
    person_found = False

    detections = []

    if results.boxes is not None and len(results.boxes) > 0:

        for box in results.boxes:

            cls_id = int(box.cls[0])
            score = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            if cls_id == PHONE_CLASS_ID and score >= phone_conf:

                phone_found = True

                detections.append(
                    f"Phone ({score:.0%})"
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (63, 50, 192),
                    3,
                )

                label = f"PHONE {score:.0%}"

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    1,
                )

                cv2.rectangle(
                    annotated,
                    (x1, max(y1 - 24, 0)),
                    (x1 + tw + 10, y1),
                    (63, 50, 192),
                    -1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            elif cls_id == BOOK_CLASS_ID and score >= book_conf:

                book_found = True

                detections.append(
                    f"Book ({score:.0%})"
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (95, 122, 29),
                    3,
                )

                label = f"BOOK {score:.0%}"

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    1,
                )

                cv2.rectangle(
                    annotated,
                    (x1, max(y1 - 24, 0)),
                    (x1 + tw + 10, y1),
                    (95, 122, 29),
                    -1,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            elif cls_id == PERSON_CLASS_ID and score >= 0.35:

                person_found = True

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (176, 148, 115),
                    1,
                )

    return (
        phone_found,
        book_found,
        person_found,
        annotated,
        detections,
    )


def _reset_phone_episode(state) -> None:
    state.phone_episode_start = None
    state.phone_last_seen = None
    state.phone_warn_tier = 0
    state.episode_handled = False


def process_logic(
    phone_detected: bool,
    book_detected: bool,
    parent_phone: str | None,
) -> None:

    state = st.session_state
    now = time.monotonic()
    kid_name = state.get("kid_name", "Champion")

    # --- 1. BOOK READING LOGIC ---
    if book_detected and not phone_detected:

        state.book_streak += 1
        _reset_phone_episode(state)

        if state.book_streak == 1:

            set_comment("good")

            record_event(
                "good",
                "Book picked up",
                f"{kid_name} picked up a book to study.",
                "success",
            )

        elif state.book_streak % 5 == 0:

            state.congratulations_count += 1
            state.total_books_read += 1

            set_comment(
                "good",
                f"Nice work, {kid_name} — that's sustained reading focus. Keep it up!",
            )

            record_event(
                "good",
                "Reading milestone achieved",
                f"{kid_name} sustained continuous reading focus.",
                "success",
            )

    # --- 2. PHONE DISTRACTION LOGIC ---
    elif phone_detected:

        state.book_streak = 0

        if state.phone_episode_start is None:

            state.phone_episode_start = now
            state.phone_warn_tier = 0
            state.episode_handled = False

        state.phone_last_seen = now

        elapsed = now - state.phone_episode_start

        warn_after = state.warn_after_seconds
        alert_after = state.alert_after_seconds

        midpoint = (
            warn_after
            + (alert_after - warn_after) / 2
        )

        if (
            state.phone_warn_tier == 0
            and elapsed >= warn_after
        ):

            state.phone_warn_tier = 1

            set_comment("bad")

            record_event(
                "bad",
                "Phone spotted",
                f"Phone in view of {kid_name} for {int(elapsed)}s.",
                "warning",
            )

        elif (
            state.phone_warn_tier == 1
            and elapsed >= midpoint
        ):

            state.phone_warn_tier = 2

            remaining = max(
                int(alert_after - elapsed),
                0,
            )

            set_comment(
                "bad",
                f"Still on the phone, {kid_name} — parent alert in about {remaining}s if this keeps up.",
            )

            record_event(
                "bad",
                "Phone distraction continuing",
                f"Phone still in view after {int(elapsed)}s.",
                "warning",
            )

        if (
            elapsed >= alert_after
            and not state.episode_handled
        ):

            state.episode_handled = True
            state.total_phone_alerts += 1

            if now - state.last_alert_at < state.cooldown_seconds:

                record_event(
                    "alert",
                    "Alert suppressed (cooldown)",
                    "SMS alert paused by cooldown timer.",
                    "info",
                )

                set_comment(
                    "bad",
                    "Distraction threshold reached, but the alert is in its cooldown period.",
                )

            else:

                state.last_alert_at = now

                if parent_phone is None:

                    state.sms_status = (
                        "Skipped: Parent phone number is missing or invalid."
                    )

                    record_event(
                        "alert",
                        "SMS skipped",
                        "Parent mobile number not configured.",
                        "warning",
                    )

                    set_comment(
                        "alert",
                        "Phone limit reached, but no parent phone number is configured for SMS.",
                    )

                else:

                    utc_time = datetime.now(
                        timezone.utc
                    ).strftime("%I:%M %p UTC")

                    body = (
                        f"Smart Study Guardian Alert: "
                        f"{kid_name} was detected using a cell phone "
                        f"during study time ({int(elapsed)}s continuous, "
                        f"at {utc_time})."
                    )

                    state.sms_status = send_sms(
                        parent_phone,
                        body,
                    )

                    record_event(
                        "alert",
                        "SMS alert sent",
                        f"Dispatched to {parent_phone}: {state.sms_status}",
                        "danger",
                    )

                    set_comment("alert")

    # --- 3. NEUTRAL / IDLE STATE ---
    else:

        if state.phone_episode_start is not None:

            since_last_seen = (
                now
                - (state.phone_last_seen or now)
            )

            if since_last_seen > state.forgiveness_seconds:

                record_event(
                    "system",
                    "Distraction cleared",
                    "Phone put away. Desk is back to neutral.",
                    "info",
                )

                set_comment(
                    "neutral",
                    f"Phone's away — ready to get back into it, {kid_name}.",
                )

                _reset_phone_episode(state)

        state.book_streak = 0


# ==============================================================================
# 6. GLOBAL PAGE CONFIG & VISUAL SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Study Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

      :root {
        --paper: #F6F5F1;
        --panel: #FFFFFF;
        --panel-sunken: #F0EFEA;
        --ink: #1C1B18;
        --ink-soft: #5C5A52;
        --ink-faint: #9B988D;
        --line: #E3E1D9;
        --line-strong: #CFCCC0;
        --accent: #1E3A8A;
        --accent-soft: #E7EBF7;
        --good: #2F6B3E;
        --good-soft: #E7F0E5;
        --warn: #9A6B12;
        --warn-soft: #F5EDDA;
        --bad: #A3312B;
        --bad-soft: #F6E7E5;
      }

      html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
      }

      .stApp {
        background: var(--paper);
        color: var(--ink);
      }

      section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--line);
      }

      h1, h2, h3, h4 {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.01em;
      }

      h5, h6 {
        color: var(--ink);
        font-weight: 600;
      }

      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
      section[data-testid="stSidebar"] details summary,
      section[data-testid="stSidebar"] .brand-subtitle {
        color: var(--ink) !important;
      }

      div[data-testid="stTopNav"],
      header[data-testid="stHeader"] {
        background: var(--panel) !important;
        border-bottom: 1px solid var(--line);
      }

      .g-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 16px;
      }

      .g-card h4,
      .g-card h5 {
        margin-top: 0;
      }

      .brand-mark {
        display: flex;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 2px;
      }

      .brand-mark .glyph {
        font-size: 18px;
      }

      .brand-title {
        font-family: 'Source Serif 4', serif;
        font-size: 19px;
        font-weight: 600;
        color: var(--ink);
      }

      .brand-subtitle {
        color: var(--ink-faint);
        font-size: 12px;
        margin-bottom: 18px;
      }

      .stat-tile {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 16px 18px;
      }

      .stat-tile .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 600;
        color: var(--ink);
        line-height: 1.15;
      }

      .stat-tile .lbl {
        color: var(--ink-faint);
        font-size: 12px;
        margin-top: 5px;
      }

      .activity-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 20px;
      }

      .activity-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--line);
      }

      .activity-head .who {
        font-weight: 600;
        font-size: 14px;
        color: var(--ink);
      }

      .activity-head .when {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--ink-faint);
      }

      .note {
        background: var(--panel-sunken);
        border-left: 3px solid var(--line-strong);
        border-radius: 3px;
        padding: 14px 16px;
        font-size: 14px;
        line-height: 1.55;
        color: var(--ink);
      }

      .note.good {
        border-left-color: var(--good);
        background: var(--good-soft);
      }

      .note.bad {
        border-left-color: var(--bad);
        background: var(--bad-soft);
      }

      .note.alert {
        border-left-color: var(--warn);
        background: var(--warn-soft);
      }

      .pill {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid transparent;
      }

      .pill-good {
        background: var(--good-soft);
        color: var(--good);
      }

      .pill-bad {
        background: var(--bad-soft);
        color: var(--bad);
      }

      .pill-alert {
        background: var(--warn-soft);
        color: var(--warn);
      }

      .pill-info {
        background: var(--accent-soft);
        color: var(--accent);
      }

      .pill-neutral {
        background: var(--panel-sunken);
        color: var(--ink-soft);
      }

      .feed-row {
        font-size: 13px;
        padding: 9px 0;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: baseline;
        gap: 10px;
      }

      .feed-row:last-child {
        border-bottom: none;
      }

      .feed-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        color: var(--ink-faint);
        white-space: nowrap;
      }

      .feed-text {
        color: var(--ink-soft);
      }

      .entry {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 3px solid var(--line-strong);
        border-radius: 3px;
        padding: 13px 16px;
        margin-bottom: 8px;
      }

      .entry.success {
        border-left-color: var(--good);
      }

      .entry.warning {
        border-left-color: var(--warn);
      }

      .entry.danger {
        border-left-color: var(--bad);
      }

      .entry.info {
        border-left-color: var(--accent);
      }

      .entry .row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
      }

      .entry .ttl {
        font-weight: 600;
        font-size: 13.5px;
        color: var(--ink);
      }

      .entry .ts {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        color: var(--ink-faint);
        white-space: nowrap;
      }

      .entry .dtl {
        font-size: 12.5px;
        color: var(--ink-soft);
        margin-top: 3px;
      }

      div.stButton > button {
        border-radius: 5px !important;
        font-weight: 500 !important;
        border: 1px solid var(--line-strong) !important;
        background: var(--panel) !important;
        color: var(--ink) !important;
      }

      div.stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 7. GLOBAL SIDEBAR
# ==============================================================================
with st.sidebar:

    st.markdown(
        '<div class="brand-mark"><span class="glyph">🛡️</span><span class="brand-title">Study Guardian</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="brand-subtitle">Study companion &amp; safety monitor</div>',
        unsafe_allow_html=True,
    )

    st.session_state.kid_name = st.text_input(
        "Student name",
        value=st.session_state.kid_name,
        placeholder="e.g. Ayan, Sara, Ali",
        help="Personalizes positive reinforcement and SMS messages.",
    )

    st.markdown("##### Live control")

    monitoring_toggle = st.toggle(
        "Active monitoring",
        value=st.session_state.monitoring,
        key="master_monitor_toggle",
    )

    if monitoring_toggle != st.session_state.monitoring:

        st.session_state.monitoring = monitoring_toggle

        if monitoring_toggle:

            record_event(
                "system",
                "Monitor started",
                "Webcam surveillance initialized.",
                "info",
            )

            set_comment(
                "neutral",
                f"Monitoring started. Ready for a productive session, {st.session_state.kid_name}.",
            )

        else:

            close_camera()

            record_event(
                "system",
                "Monitor paused",
                "Surveillance stopped by user.",
                "info",
            )

            set_comment(
                "neutral",
                "Monitoring paused.",
            )

    c_btn1, c_btn2 = st.columns(2)

    with c_btn1:

        if st.button(
            "Reset counts",
            use_container_width=True,
        ):

            st.session_state.book_streak = 0

            _reset_phone_episode(
                st.session_state
            )

            record_event(
                "system",
                "Counts reset",
                "Distraction and reading counts reset manually.",
                "info",
            )

            set_comment(
                "neutral",
                "Detection streak reset.",
            )

            st.rerun()

    with c_btn2:

        if st.button(
            "Test milestone",
            use_container_width=True,
        ):

            st.session_state.congratulations_count += 1
            st.session_state.total_books_read += 1

            set_comment("good")

            st.rerun()

    with st.expander("Behavioral testing"):

        st.caption(
            "Trigger app behavior without needing physical items in front of the webcam."
        )

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:

            if st.button(
                "Simulate: reading",
                use_container_width=True,
            ):

                process_logic(
                    phone_detected=False,
                    book_detected=True,
                    parent_phone=normalize_pk_mobile(
                        st.session_state.parent_phone
                    ),
                )

                st.rerun()

        with sim_col2:

            if st.button(
                "Simulate: phone",
                use_container_width=True,
            ):

                process_logic(
                    phone_detected=True,
                    book_detected=False,
                    parent_phone=normalize_pk_mobile(
                        st.session_state.parent_phone
                    ),
                )

                st.rerun()

    st.markdown("---")

    health_text = (
        "Live"
        if st.session_state.monitoring
        else "Standby"
    )

    st.caption(
        f"Status: **{health_text}** · Alert after: **{int(st.session_state.alert_after_seconds)}s**"
    )


# ==============================================================================
# 8. PAGE FUNCTIONS
# ==============================================================================
def page_live_monitor() -> None:

    bar_col1, bar_col2, bar_col3, bar_col4 = st.columns(4)

    with bar_col1:

        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="val">
                    {'ACTIVE' if st.session_state.monitoring else 'PAUSED'}
                </div>
                <div class="lbl">System state</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bar_col2:

        if st.session_state.phone_episode_start is not None:

            elapsed_s = int(
                time.monotonic()
                - st.session_state.phone_episode_start
            )

            phone_fraction = (
                f"{elapsed_s}s/"
                f"{int(st.session_state.alert_after_seconds)}s"
            )

            color = "var(--bad)"

        else:

            phone_fraction = (
                f"0s/"
                f"{int(st.session_state.alert_after_seconds)}s"
            )

            color = "var(--ink)"

        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="val" style="color:{color}">
                    {phone_fraction}
                </div>
                <div class="lbl">Continuous phone time</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bar_col3:

        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="val" style="color:var(--good)">
                    {st.session_state.total_books_read}
                </div>
                <div class="lbl">Reading milestones</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bar_col4:

        color = (
            "var(--warn)"
            if st.session_state.total_phone_alerts > 0
            else "var(--ink)"
        )

        st.markdown(
            f"""
            <div class="stat-tile">
                <div class="val" style="color:{color}">
                    {st.session_state.total_phone_alerts}
                </div>
                <div class="lbl">Alerts dispatched</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height: 14px;'></div>",
        unsafe_allow_html=True,
    )

    col_cam, col_activity = st.columns(
        [13, 8],
        gap="medium",
    )

    with col_cam:

        st.markdown(
            '<div class="g-card">',
            unsafe_allow_html=True,
        )

        st.markdown("#### Desk feed")

        @st.fragment(
            run_every=(
                st.session_state.feed_refresh_seconds
                if st.session_state.monitoring
                else None
            )
        )
        def live_camera_fragment() -> None:

            state = st.session_state

            if not state.monitoring:

                close_camera()

                st.info(
                    "Turn on **Active monitoring** in the sidebar to start the webcam stream."
                )

                return

            if state.camera is None:

                cam = cv2.VideoCapture(
                    state.camera_index
                )

                cam.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    640,
                )

                cam.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    480,
                )

                cam.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1,
                )

                state.camera = cam
                state.last_check_at = float("-inf")

            if not state.camera.isOpened():

                close_camera()

                st.error(
                    "Could not open the webcam device. Check camera permissions or pick another camera in Settings."
                )

                return

            ok, frame = state.camera.read()

            if not ok:

                close_camera()

                st.error(
                    "Failed to capture a camera frame. Toggle monitoring to retry."
                )

                return

            now = time.monotonic()

            display_frame = frame

            if now - state.last_check_at >= state.check_interval:

                state.last_check_at = now

                try:

                    phone_found, book_found, person_found, annotated, dets = detect_objects(
                        frame,
                        state.phone_conf,
                        state.book_conf,
                        resolve_model_path(
                            state.model_family,
                            state.model_variant,
                        ),
                        state.infer_size,
                    )

                    display_frame = annotated

                    parent_phone = normalize_pk_mobile(
                        state.parent_phone
                    )

                    process_logic(
                        phone_found,
                        book_found,
                        parent_phone,
                    )

                except Exception as e:

                    st.warning(
                        f"Detection warning: {e}"
                    )

            h, w = display_frame.shape[:2]

            max_w = 640

            if w > max_w:

                scale = max_w / w

                display_frame = cv2.resize(
                    display_frame,
                    (
                        max_w,
                        int(h * scale),
                    ),
                )

            st.image(
                cv2.cvtColor(
                    display_frame,
                    cv2.COLOR_BGR2RGB,
                ),
                channels="RGB",
                use_container_width=True,
            )

            if state.phone_episode_start is not None:

                elapsed = (
                    time.monotonic()
                    - state.phone_episode_start
                )

                progress = min(
                    elapsed
                    / state.alert_after_seconds,
                    1.0,
                )

                st.progress(
                    progress,
                    text=(
                        f"Distraction warning: "
                        f"{int(elapsed)}s of "
                        f"{int(state.alert_after_seconds)}s "
                        f"continuous phone use"
                    ),
                )

            elif state.book_streak > 0:

                st.success(
                    f"{state.kid_name} is actively reading — "
                    f"focus streak: {state.book_streak} checks"
                )

        live_camera_fragment()

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    with col_activity:

        comment = st.session_state.last_comment

        c_type = comment.get(
            "type",
            "neutral",
        )

        c_badge = comment.get(
            "badge",
            "Standby",
        )

        c_text = comment.get(
            "text",
            "",
        )

        c_time = comment.get(
            "time",
            "",
        )

        pill_class = (
            c_type
            if c_type in ("good", "bad", "alert")
            else "neutral"
        )

        st.markdown(
            f"""
            <div class="activity-panel">
              <div class="activity-head">
                <span class="who">Activity monitor</span>
                <span class="when">{c_time} UTC</span>
              </div>

              <div style="margin-bottom: 10px;">
                <span class="pill pill-{pill_class}">
                    {c_badge}
                </span>
              </div>

              <div class="note {c_type}">
                {c_text}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='height: 14px;'></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="g-card">',
            unsafe_allow_html=True,
        )

        st.markdown("##### Recent activity")

        if st.session_state.comment_history:

            for item in list(
                st.session_state.comment_history
            )[:6]:

                st.markdown(
                    f"""
                    <div class="feed-row">
                        <span class="feed-time">
                            {item.get('time')}
                        </span>
                        <span class="feed-text">
                            {item.get('text')}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.caption(
                "No activity logged yet. Start monitoring to see updates here."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )


def page_activity_log() -> None:

    st.markdown(
        f"""
        <span style='color: var(--ink-faint);'>
            Session history and behavioral timeline for
            <b>{st.session_state.kid_name}</b>.
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height: 10px;'></div>",
        unsafe_allow_html=True,
    )

    good_events = sum(
        1
        for e in st.session_state.events
        if e["category"] == "good"
    )

    bad_events = sum(
        1
        for e in st.session_state.events
        if e["category"] == "bad"
    )

    alert_events = sum(
        1
        for e in st.session_state.events
        if e["category"] == "alert"
    )

    focus_score = 100

    if good_events + bad_events > 0:

        focus_score = int(
            (
                good_events
                / (good_events + bad_events)
            )
            * 100
        )

    score_color = (
        "var(--good)"
        if focus_score >= 80
        else (
            "var(--warn)"
            if focus_score >= 50
            else "var(--bad)"
        )
    )

    kpi_specs = [
        (
            f"{focus_score}%",
            "Session focus score",
            score_color,
        ),
        (
            str(good_events),
            "Reading praise events",
            "var(--good)",
        ),
        (
            str(bad_events),
            "Phone distractions",
            "var(--bad)",
        ),
        (
            str(alert_events),
            "SMS alerts sent",
            "var(--warn)",
        ),
    ]

    cols = st.columns(4)

    for col, (val, lbl, color) in zip(
        cols,
        kpi_specs,
    ):

        with col:

            st.markdown(
                f"""
                <div class="g-card" style="text-align:center;">
                    <div style="
                        font-family:'JetBrains Mono',monospace;
                        font-size:24px;
                        font-weight:600;
                        color:{color};
                    ">
                        {val}
                    </div>

                    <div class="lbl" style="margin-top:4px;">
                        {lbl}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div style='height: 6px;'></div>",
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2, filter_col3 = st.columns(
        [2, 3, 2]
    )

    with filter_col1:

        event_filter = st.selectbox(
            "Filter category",
            [
                "All events",
                "Good behaviors (books)",
                "Distractions (phones)",
                "Parent alerts",
                "System info",
            ],
        )

    with filter_col2:

        search_query = st.text_input(
            "Search event descriptions",
            placeholder="Filter by keyword…",
        )

    with filter_col3:

        st.markdown(
            "<div style='height: 28px;'></div>",
            unsafe_allow_html=True,
        )

        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:

            if st.session_state.events:

                df = pd.DataFrame(
                    st.session_state.events
                )

                csv = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Export CSV",
                    data=csv,
                    file_name=(
                        f"study_guardian_audit_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )

        with c_exp2:

            if st.button(
                "Clear log",
                use_container_width=True,
            ):

                st.session_state.events = []

                st.rerun()

    st.markdown(
        "##### Chronological event log"
    )

    if not st.session_state.events:

        st.info(
            "No activity logged yet. Events will appear here once monitoring is active."
        )

    else:

        for ev in st.session_state.events:

            cat = ev["category"]

            if (
                event_filter
                == "Good behaviors (books)"
                and cat != "good"
            ):
                continue

            if (
                event_filter
                == "Distractions (phones)"
                and cat != "bad"
            ):
                continue

            if (
                event_filter
                == "Parent alerts"
                and cat != "alert"
            ):
                continue

            if (
                event_filter
                == "System info"
                and cat != "system"
            ):
                continue

            if search_query:

                q = search_query.lower()

                if (
                    q not in ev["title"].lower()
                    and q not in ev["details"].lower()
                ):
                    continue

            sev = ev.get(
                "severity",
                "info",
            )

            st.markdown(
                f"""
                <div class="entry {sev}">
                  <div class="row">
                    <span class="ttl">
                        {ev['title']}
                    </span>

                    <span class="ts">
                        {ev['timestamp']}
                    </span>
                  </div>

                  <div class="dtl">
                    {ev['details']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_settings() -> None:

    st.markdown(
        "<span style='color: var(--ink-faint);'>Detection sensitivity, parent contact, and SMS delivery.</span>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='height: 10px;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="g-card">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Parent contact")

    phone_input = st.text_input(
        "Parent mobile number",
        value=st.session_state.parent_phone,
        placeholder="03XXXXXXXXX (Pakistan) or +923XXXXXXXXX",
        help="Pakistani local format (03001234567) or international E.164 (+923001234567 / +1234567890).",
    )

    st.session_state.parent_phone = phone_input

    norm = normalize_pk_mobile(phone_input)

    if phone_input:

        if norm:

            st.success(
                f"Verified phone destination: **{norm}**"
            )

        else:

            st.warning(
                "Invalid mobile format. Enter a valid Pakistani number "
                "(e.g. 03001234567) or international E.164 number."
            )

    else:

        st.info(
            "Enter a parent mobile number to enable SMS distraction alerts."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="g-card">',
        unsafe_allow_html=True,
    )

    st.markdown("#### SMS gateway (Twilio)")

    st.caption(
        "Provide credentials here, or set TWILIO_ACCOUNT_SID, "
        "TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER in the environment."
    )

    tw_col1, tw_col2, tw_col3 = st.columns(3)

    with tw_col1:

        st.session_state.twilio_sid = st.text_input(
            "Account SID",
            value=st.session_state.get(
                "twilio_sid",
                "",
            ),
            type="password",
        )

    with tw_col2:

        st.session_state.twilio_token = st.text_input(
            "Auth token",
            value=st.session_state.get(
                "twilio_token",
                "",
            ),
            type="password",
        )

    with tw_col3:

        st.session_state.twilio_from = st.text_input(
            "From number",
            value=st.session_state.get(
                "twilio_from",
                "",
            ),
            placeholder="+1XXXXXXXXXX",
        )

    if st.button(
        "Send test SMS",
        help="Sends a test text message to the verified parent phone.",
    ):

        if not norm:

            st.error(
                "Enter and verify a valid parent mobile number first."
            )

        else:

            test_body = (
                f"Smart Study Guardian Test: "
                f"SMS alerts for {st.session_state.kid_name} "
                f"are operational!"
            )

            status = send_sms(
                norm,
                test_body,
            )

            if "successfully" in status:

                st.success(status)

            else:

                st.error(status)

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="g-card">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Model & performance")

    st.caption(
        "YOLOv6n is configured as the default model."
    )

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:

        variant_labels = {
            "n": "Nano — YOLOv6n",
            "s": "Small — YOLOv6s",
            "m": "Medium — YOLOv6m",
        }

        chosen_label = st.selectbox(
            "Model size",
            list(variant_labels.values()),
            index=list(
                variant_labels.keys()
            ).index(
                st.session_state.model_variant
            ),
            help="Nano is the fastest option.",
        )

        st.session_state.model_variant = next(
            k
            for k, v in variant_labels.items()
            if v == chosen_label
        )

    with perf_col2:

        st.session_state.infer_size = st.select_slider(
            "Inference resolution (px)",
            options=[
                320,
                416,
                480,
                640,
            ],
            value=(
                st.session_state.infer_size
                if st.session_state.infer_size
                in (
                    320,
                    416,
                    480,
                    640,
                )
                else 480
            ),
            help="Lower resolution = faster inference, but small or distant objects get harder to detect.",
        )

    st.caption(
        f"Currently running: **YOLOv6-{st.session_state.model_variant}** "
        f"at **{st.session_state.infer_size}px**. "
        "Changing this reloads the model on the next detection cycle."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="g-card">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Detection sensitivity")

    sens_col1, sens_col2 = st.columns(2)

    with sens_col1:

        st.session_state.phone_conf = st.slider(
            "Phone detection confidence",
            0.15,
            0.90,
            st.session_state.phone_conf,
            0.05,
            help="Higher values reduce false positives; lower values detect phones faster.",
        )

    with sens_col2:

        st.session_state.book_conf = st.slider(
            "Book detection confidence",
            0.15,
            0.90,
            st.session_state.book_conf,
            0.05,
            help="Confidence threshold required to recognize a reading book.",
        )

    st.caption(
        "Timing is based on how long the phone stays continuously in view — not on a raw number of camera checks."
    )

    sens_col3, sens_col4, sens_col5 = st.columns(3)

    with sens_col3:

        st.session_state.warn_after_seconds = st.slider(
            "Warn after (seconds)",
            2,
            30,
            int(
                st.session_state.warn_after_seconds
            ),
            1,
            help="How long the phone must stay continuously in view before the first gentle nudge appears.",
        )

    with sens_col4:

        st.session_state.alert_after_seconds = st.slider(
            "Alert after (seconds)",
            5,
            120,
            int(
                st.session_state.alert_after_seconds
            ),
            5,
            help="How long the phone must stay continuously in view before a parent SMS is dispatched.",
        )

    with sens_col5:

        st.session_state.forgiveness_seconds = st.slider(
            "Forgiveness window (seconds)",
            1,
            15,
            int(
                st.session_state.forgiveness_seconds
            ),
            1,
            help="If the phone briefly leaves view, the episode continues instead of resetting.",
        )

    if (
        st.session_state.alert_after_seconds
        <= st.session_state.warn_after_seconds
    ):

        st.warning(
            "'Alert after' should be greater than 'Warn after' for the escalation to make sense."
        )

    feed_col1, feed_col2 = st.columns(2)

    with feed_col1:

        st.session_state.feed_refresh_seconds = st.slider(
            "Live feed refresh rate (seconds)",
            0.05,
            0.5,
            st.session_state.feed_refresh_seconds,
            0.05,
            help="How often the video preview updates on screen.",
        )

    with feed_col2:

        st.session_state.check_interval = st.slider(
            "Detection sample rate (seconds)",
            0.5,
            3.0,
            st.session_state.check_interval,
            0.25,
            help="How often the model actually looks at the webcam frame.",
        )

    st.session_state.camera_index = st.selectbox(
        "Webcam device index",
        [0, 1, 2],
        index=st.session_state.camera_index,
    )

    st.session_state.cooldown_seconds = st.slider(
        "Alert cooldown (minutes)",
        1,
        30,
        int(
            st.session_state.cooldown_seconds / 60
        ),
        1,
    ) * 60

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# 9. NAVIGATION
# ==============================================================================
nav = st.navigation(
    [
        st.Page(
            page_live_monitor,
            title="Live Monitor",
            icon="📡",
            url_path="live-monitor",
            default=True,
        ),
        st.Page(
            page_activity_log,
            title="Activity Log",
            icon="🗂️",
            url_path="activity-log",
        ),
        st.Page(
            page_settings,
            title="Settings",
            icon="⚙️",
            url_path="settings",
        ),
    ],
    position="top",
)

nav.run()
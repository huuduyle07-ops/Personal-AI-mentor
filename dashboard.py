"""Local web dashboard for the Korean study assistant.

    uv run --extra dashboard streamlit run dashboard.py
    uv run --extra dashboard streamlit run dashboard.py -- --db path/to/other.db --user 123

Database: --db, else DASHBOARD_DB, else DATABASE_PATH (default data/accountability.db).
User: --user, else DASHBOARD_USER, else ALLOWED_USER_ID (your Discord data), else 1 (the local chat user).
Streamlit is bound to localhost and usage stats are off (.streamlit/config.toml).

The Overview, Memory and Check-in tabs only read. The Practice tab writes only when you press a button:
quiz answers go through the same code as the chat, and note words are added only after "Add to my deck".
"""
import argparse
import asyncio
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from active_quiz import feedback
from config import Config
from dashboard_data import ChatSession, Dashboard, context_rows, short_day, voice_message, when
from speech import SpeechUnavailable, Transcriber
from study_notes import IMAGE_TYPES, TEXT_TYPES, NoteImporter, VisionUnavailable
from time_utils import now_kst

load_dotenv(".env")


def settings():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db")
    parser.add_argument("--user", type=int)
    args, _ = parser.parse_known_args()
    config = Config.from_env()
    db = args.db or os.environ.get("DASHBOARD_DB") or config.database_path
    user = args.user or int(os.environ.get("DASHBOARD_USER") or 0) or config.allowed_user_id or 1
    return config, db, user


CONFIG, DB, USER = settings()
MIC_LANGUAGES = {"Auto": None, "English": "en", "한국어": "ko", "Tiếng Việt": "vi"}
st.set_page_config(page_title="Korean study dashboard", page_icon="📚", layout="wide")


@st.cache_resource
def data(path, user):
    return Dashboard(path, user)


@st.cache_resource
def chat_session(path, user):
    return ChatSession(CONFIG, path, user)


@st.cache_resource
def transcriber():
    return Transcriber()


board = data(DB, USER)
NOW = now_kst()
TODAY = NOW.date().isoformat()

st.title("Korean study dashboard")
st.caption(f"Database: `{DB}` · user {USER} · {NOW:%a %b %d, %H:%M} KST")
chat_tab, overview_tab, memory_tab, policy_tab, practice_tab = st.tabs(
    ["Chat", "Overview & Focus", "Memory & Confusion Pairs", "Check-in Policy (LinUCB)", "Practice & Note Import"])

# ---- Chat ------------------------------------------------------------------------------------
# Runs first, so the other tabs already show anything a message just recorded.
with chat_tab:
    st.caption("Talk to the bot exactly as in the Discord DM or the terminal chat: same conversation logic, same "
               "drafts and confirmations, same database. Commands like /focus, /recall or /quiz active work too.")
    history = st.session_state.setdefault("chat", [])
    if history and len(history[0]) == 2:  # messages from before pictures were supported
        history[:] = [(role, text, None) for role, text in history]
    shortcuts = ["how am I doing?", "quiz me", "quiz me in Korean", "what should I focus on today?"]
    pressed = None
    for column, text in zip(st.columns(len(shortcuts) + 1), shortcuts + ["Clear chat"]):
        if column.button(text, key=f"shortcut_{text}", width="stretch"):
            pressed = text
    if pressed == "Clear chat":
        history.clear()
        pressed = None
    box = st.container(height=480)  # the conversation, above the input box
    voice = transcriber()
    if not voice.ready():
        st.caption(f"🎤 Voice input runs locally with Whisper “{voice.model_name}”. It needs a one-time download "
                   f"of about {voice.size}; after that nothing leaves your computer.")
        if st.button(f"Download the speech model ({voice.size})", key="download_voice"):
            with st.spinner("Downloading the speech model… this can take a few minutes"):
                try:
                    voice.download()
                    st.rerun()
                except SpeechUnavailable as error:
                    st.warning(str(error))
    mic_language = st.segmented_control("🎤 I'm speaking", list(MIC_LANGUAGES), default="Auto",
                                        key="mic_language", help="Auto tells English and Korean apart by itself; "
                                        "pick a language if it keeps guessing wrong.") or "Auto"
    submitted = st.chat_input("Message the bot, attach a photo of your notes, or press the mic and speak",
                              accept_file=True, file_type=sorted(t.lstrip(".") for t in TEXT_TYPES | IMAGE_TYPES),
                              accept_audio=True)
    message = (submitted.text if submitted else None) or pressed or ""  # text is None for a voice-only message
    files = list(submitted.files) if submitted else []
    audio = submitted.get("audio") if submitted else None
    shown = message  # what the user's chat bubble says
    if audio is not None:
        with st.spinner("Listening…"):
            allow_vietnamese, hints = board.listening_hints(TODAY)
            message, shown, note = voice_message(voice, message, audio.getvalue(), MIC_LANGUAGES[mic_language],
                                                 hints, allow_vietnamese)
        if note:
            history.append(("assistant", note, None))
    if message or files:
        session = chat_session(DB, USER)
        with st.spinner("Reading your notes…" if files else "Thinking…"):
            for upload in files:
                data = upload.getvalue()
                image = data if upload.name.lower().endswith(tuple(IMAGE_TYPES)) else None
                history.append(("user", f"📎 {upload.name}", image))
                try:
                    history.append(("assistant", session.send_file(upload.name, data), None))
                except Exception as error:  # Ollama down, timeout: show it, keep the page working
                    history.append(("assistant", f"Sorry, I couldn't read that file ({type(error).__name__}).", None))
            if message:
                history.append(("user", shown, None))
                try:
                    history.append(("assistant", session.send(message), None))
                except Exception as error:
                    history.append(("assistant", f"Sorry, something went wrong ({type(error).__name__}). Try again.", None))
    if not history:
        box.info("Say hi, log your study time (“today's total is 20 minutes”), ask for a quiz, or attach a photo "
                 "of your notes to add its words to your deck (you confirm with “yes”).")
    for role, text, image in history:
        with box.chat_message(role):
            if image is not None:
                st.image(image, width=240)
            st.markdown(text.replace("\n", "  \n"))

# ---- Overview & Focus ------------------------------------------------------------------------
with overview_tab:
    o = board.overview(NOW)
    personalized = o["model"] == "personalized"
    s = o["shadow"]
    cards = [
        ("Today's minutes", "—" if o["minutes"] is None else o["minutes"],
         "no goal yet" if o["target"] is None else f"target {o['target']} min"),
        ("Streak", f"{o['streak']} day{'s' if o['streak'] != 1 else ''}", "days in a row meeting the goal"),
        ("Due words", o["due"], f"{o['resting']} resting · {o['seen']} of {o['deck']} seen"),
        ("Recall model", "Personalized" if personalized else "Population",
         f"from {o['reviews']} reviews" if personalized else f"{o['reviews']} of {o['min_events']} reviews to personalize"),
        ("Shadow windows", f"{s['scored']} scored", f"{s['waiting']} waiting"),
    ]
    for col, (label, value, note) in zip(st.columns(5), cards):
        col.metric(label, value)
        col.caption(note)
    st.info(o["suggestion"])
    st.subheader("Last 7 days")
    week = pd.DataFrame(o["week"])
    week["label"] = week["day"].map(short_day)
    # Categorical axis in date order (a plain string axis would sort the labels alphabetically).
    st.bar_chart(week, x="label", y="minutes", x_label="", y_label="minutes (latest reported total)", sort=False)
    if week["target"].notna().any():
        st.caption("Daily targets: " + ", ".join(f"{r.label} {int(r.target)}" for r in week.itertuples()
                                                  if pd.notna(r.target)))
    with st.expander("Focus summary (same text as /focus)"):
        st.text(o["text"])

# ---- Memory & Confusion Pairs ----------------------------------------------------------------
with memory_tab:
    m = board.memory(NOW)
    left, right = st.columns([2, 1])
    with left:
        mode = "personalized weights" if m["personalized"] else "population prior (Duolingo)"
        st.markdown(f"**Model in use:** {mode}. {m['review_events']} genuine review events "
                    f"({m['correct_events']} correct); personalization starts at {m['min_events']}.")
        st.progress(min(m["review_events"] / m["min_events"], 1.0),
                    text=f"{min(m['review_events'], m['min_events'])} of {m['min_events']} reviews to personalize")
    with right:
        st.dataframe(pd.DataFrame(m["weights"]).set_index("feature"), width="stretch")
    st.subheader(f"Words seen: {m['seen']} of {m['deck']}")
    if m["words"]:
        words = pd.DataFrame(m["words"])
        statuses = st.multiselect("Show", ["due", "resting", "not due"], default=["due", "resting", "not due"])
        shown = words[words["status"].isin(statuses)]
        st.bar_chart(shown, x="word", y="recall", color="status", y_label="predicted recall", x_label="")
        st.dataframe(shown.drop(columns="half_life_days").assign(recall=shown["recall"].round(3)),
                     hide_index=True, width="stretch",
                     column_config={"recall": st.column_config.ProgressColumn("recall", min_value=0, max_value=1,
                                                                              format="%.2f")})
    else:
        st.write("No quiz answers yet. Try the Practice tab.")
    if m["next"]:
        st.caption(f"Next flashcard: {m['next']['word']} ({m['next']['policy'].replace('_', ' ')})")
    st.subheader("Confusion pairs")
    if m["pairs"]:
        st.dataframe(pd.DataFrame([{"missed": p["word"], "meaning": p["meaning"], "mixed up with": p["other"],
                                    "its meaning": p["other_meaning"], "evidence": p["kind"], "count": p["count"],
                                    "jamo similarity": round(p["similarity"], 2)} for p in m["pairs"]]),
                     hide_index=True, width="stretch")
    else:
        st.write("No mix-ups detected yet. They come from wrong flashcard answers that are another word's meaning, "
                 "another word typed in an active quiz, or look-alike words missed twice.")

# ---- Check-in Policy -------------------------------------------------------------------------
with policy_tab:
    p = board.policy(NOW)
    c = p["checkins"]
    state = "off" if not c or not c["enabled"] else ("paused" if c["paused"] else f"on at {c['midday']} and {c['evening']}")
    st.markdown(f"**Would choose now:** `{p['action']}` · check-ins {state} · α = {p['alpha']} · "
                f"{p['updates']} learned updates · preview only, nothing is sent.")
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Context")
        st.dataframe(pd.DataFrame(context_rows(p["features"])).set_index("feature"), width="stretch")
    with right:
        st.subheader("Score per action")
        scores = pd.DataFrame(p["scores"])
        long = scores.melt(id_vars="action", value_vars=["predicted_reward", "exploration_bonus"],
                           var_name="part", value_name="value")
        # Side by side: the predicted reward can be negative, which a stacked bar hides.
        st.bar_chart(long, x="action", y="value", color="part", stack=False, sort=False, x_label="",
                     y_label="score (UCB = reward + bonus)")
        st.dataframe(scores.round(3).set_index("action"), width="stretch")
    st.subheader("Shadow-mode history")
    st.caption(f"{p['shadow']['learned']} scored, {p['shadow']['waiting']} waiting. A window is scored 3 hours "
               "after the check-in (or the planned 18:00 silence): engaged = studied or answered a quiz.")
    if p["history"]:
        history = pd.DataFrame(p["history"])
        history["decided_at"] = history["decided_at"].map(when)
        history["resolved_at"] = history["resolved_at"].map(when)
        st.dataframe(history, hide_index=True, width="stretch")
    else:
        st.write("No shadow windows yet; they start once check-ins are on.")

# ---- Practice & Note Import ------------------------------------------------------------------
with practice_tab:
    st.caption("Buttons here write to the database, exactly like the chat. Active answers are kept apart "
               "from flashcards, so they never change the recall model.")
    passive_col, active_col = st.columns(2)
    waiting = board.pending_quizzes(TODAY)

    with passive_col:
        st.subheader("Flashcard: Korean → English")
        task = waiting["passive"]
        if task is None:
            if st.button("Start a flashcard", key="start_passive"):
                try:
                    board.start_passive(TODAY)
                    st.rerun()
                except ValueError as error:
                    st.warning(f"{error} (flashcards check today's study, so log minutes in the chat first).")
        else:
            with st.form("passive_answer", clear_on_submit=True):
                st.markdown(f"### {task['word']}")
                typed = st.text_input("English meaning")
                if st.form_submit_button("Check") and typed.strip():
                    try:
                        st.session_state["passive_result"] = board.answer_passive(TODAY, task["id"], typed)
                        st.rerun()
                    except ValueError as error:
                        st.warning(str(error))
        result = st.session_state.pop("passive_result", None)
        if result:
            (st.success if result["correct"] else st.error)(
                f"{'Correct' if result['correct'] else 'Not quite'}: {result['word']} = {result['meaning']}")

    with active_col:
        st.subheader("Active: English → Korean")
        task = waiting["active"]
        if task is None:
            if st.button("Start an active quiz", key="start_active"):
                try:
                    board.start_active(TODAY)
                    st.rerun()
                except ValueError as error:
                    st.warning(str(error))
        else:
            with st.form("active_answer", clear_on_submit=True):
                st.markdown(f"### {task['meaning']}")
                typed = st.text_input("Type it in Hangul (spaces and a final period are fine)")
                if st.form_submit_button("Check") and typed.strip():
                    try:
                        st.session_state["active_result"] = board.answer_active(TODAY, task["id"], typed)
                        st.rerun()
                    except ValueError as error:
                        st.warning(str(error))
        result = st.session_state.pop("active_result", None)
        if result:
            (st.success if result["correct"] else st.error)(feedback(result).split(" Say “")[0])

    st.divider()
    st.subheader("Import words from your notes")
    st.caption(f"Text files use fixed parsing rules; photos are read by the local vision model "
               f"`{CONFIG.ollama_vision_model}`. Nothing is saved until you press “Add to my deck”.")
    upload = st.file_uploader("Notes file or photo", type=sorted(t.lstrip(".") for t in TEXT_TYPES | IMAGE_TYPES))
    if upload is not None and st.button("Extract words", key="extract"):
        with st.spinner("Reading your notes…"):
            try:
                pairs, source = asyncio.run(NoteImporter(CONFIG).extract(upload.name, upload.getvalue()))
                items, skipped = board.preview_import(pairs)
                st.session_state["import"] = {"items": items, "skipped": skipped, "source": source}
            except (ValueError, VisionUnavailable) as error:
                st.session_state.pop("import", None)
                st.warning(str(error))
    proposal = st.session_state.get("import")
    if proposal:
        if proposal["skipped"]:
            st.caption("Already in your deck, skipped: " + ", ".join(proposal["skipped"]))
        if not proposal["items"]:
            st.write(f"No new Korean–English pairs found in your {proposal['source']}.")
        else:
            table = pd.DataFrame([{"add": True, "korean": word, "english": ", ".join(meanings)}
                                  for word, meanings in proposal["items"]])
            edited = st.data_editor(table, hide_index=True, disabled=["korean", "english"], width="stretch",
                                    key="import_table")
            chosen = [proposal["items"][i] for i in edited.index[edited["add"]]]
            add, cancel = st.columns(2)
            if add.button(f"Add to my deck ({len(chosen)} words)", type="primary", disabled=not chosen):
                st.success(board.confirm_import(chosen, proposal["source"], TODAY))
                st.session_state.pop("import", None)
            if cancel.button("Cancel"):
                st.session_state.pop("import", None)
                st.rerun()

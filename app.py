"""
Intern Task Bot
---------------
Flow:
 1. HR/Manager creates a task (POST /tasks) with the company's EXPECTED OUTPUT.
 2. Claude explains the task in a human voice in Tamil, English, Hindi -> emailed to intern
    along with the submission link and review date.
 3. Intern submits via the link (text or file).
 4. On review date a scheduler compares submission vs expected output using Claude.
      - match    -> task COMPLETED, congrats mail, loop stops.
      - no match -> feedback (3 languages) + what to fix + NEW review date + same link, loop continues.
      - nothing submitted -> reminder + new review date.
Run:  pip install fastapi uvicorn apscheduler anthropic python-multipart
      export ANTHROPIC_API_KEY=... SMTP_HOST=... SMTP_USER=... SMTP_PASS=... BASE_URL=http://localhost:8000
      uvicorn app:app --reload
"""
import asyncio, json, os, re, smtplib, sqlite3, tempfile, uuid
from datetime import datetime, timedelta
from email.message import EmailMessage

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import anthropic
import edge_tts
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

DB = os.getenv("DB_PATH", "interns.db")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_FROM = os.getenv("MAIL_FROM", SMTP_USER)
FMT = "%Y-%m-%d %H:%M"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"   # 1 = no Claude calls, fake AI replies (free testing)
CHECK_MINUTES = int(os.getenv("REVIEW_CHECK_MINUTES", "10"))

client = None if DRY_RUN else anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
app = FastAPI(title="Intern Task Bot")


# ---------- DB ----------
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(
          id INTEGER PRIMARY KEY AUTOINCREMENT, token TEXT UNIQUE,
          intern_name TEXT, intern_email TEXT, title TEXT, description TEXT,
          expected_output TEXT, status TEXT DEFAULT 'active',
          iteration INTEGER DEFAULT 1, review_days INTEGER DEFAULT 2, review_date TEXT);
        CREATE TABLE IF NOT EXISTS submissions(
          id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, iteration INTEGER,
          content TEXT, created TEXT, evaluated INTEGER DEFAULT 0);
        """)


init_db()


# ---------- Claude helpers ----------
def ask_claude(system: str, user: str, max_tokens=2000) -> str:
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in r.content if b.type == "text")


def ask_json(system: str, user: str) -> dict:
    txt = ask_claude(system + "\nReturn ONLY valid JSON. No markdown fences.", user)
    return json.loads(txt.replace("```json", "").replace("```", "").strip())


TONE = ("You are a friendly, warm senior mentor at the company talking to a new intern. "
        "Write like a real human, not a robot: simple words, encouraging, short sentences. "
        "Tamil must be in Tamil script (natural spoken-style Tamil), Hindi in Devanagari, English in plain English. "
        "Do not translate word-by-word; write each language naturally.")


def explain_task(t) -> dict:
    if DRY_RUN:
        return {
            "en": f"[MOCK] Hi {t['intern_name']}! Your task is '{t['title']}'. {t['description']}. Expected output: {t['expected_output']}.",
            "ta": f"[MOCK] வணக்கம் {t['intern_name']}! உங்கள் task: '{t['title']}'. எதிர்பார்க்கும் output: {t['expected_output']}.",
            "hi": f"[MOCK] नमस्ते {t['intern_name']}! आपका टास्क: '{t['title']}'. अपेक्षित आउटपुट: {t['expected_output']}.",
        }
    return ask_json(TONE, f"""Explain this task to intern {t['intern_name']}.
Title: {t['title']}
Description: {t['description']}
Expected output (explain what 'done' looks like, without being vague): {t['expected_output']}
JSON keys: en, ta, hi (each a friendly explanation with: what to do, what final output we expect, tips).""")


def evaluate(t, submission: str) -> dict:
    if DRY_RUN:  # test rule: submission containing the word "final" = match, anything else = not matching
        ok = "final" in submission.lower()
        return {
            "match": ok,
            "done_well": ["[MOCK] You submitted on time"],
            "needs_fix": [] if ok else ["[MOCK] Missing the final result - include the word 'final' to pass this test"],
            "feedback_en": "[MOCK] Great, this matches the expected output!" if ok else "[MOCK] Good start, but it does not match the expected output yet. Please fix and resubmit.",
            "feedback_ta": "[MOCK] சூப்பர்! Expected output-உடன் பொருந்துகிறது." if ok else "[MOCK] நல்ல தொடக்கம், ஆனா இன்னும் expected output-உடன் பொருந்தல. திருத்தி மறுபடி submit பண்ணுங்க.",
            "feedback_hi": "[MOCK] बहुत बढ़िया, यह अपेक्षित आउटपुट से मेल खाता है!" if ok else "[MOCK] अच्छी शुरुआत, पर अभी आउटपुट मेल नहीं खाता। सुधारकर फिर से सबमिट करें।",
        }
    return ask_json(TONE + " You are also a fair reviewer.", f"""Compare the intern's submission with the company's expected output.
Task: {t['title']} - {t['description']}
EXPECTED OUTPUT: {t['expected_output']}
INTERN SUBMISSION (attempt #{t['iteration']}): {submission[:30000]}

Decide match=true ONLY if the submission fulfils the expected output (minor style differences are fine).
JSON keys:
 match (bool), done_well (list of short strings), needs_fix (list of short, specific strings),
 feedback_en, feedback_ta, feedback_hi (human-voice feedback: appreciate what is good, clearly say what to change and why, encourage).""")


# ---------- Voice (text -> speech, free, no API key) ----------
VOICES = {"ta": "ta-IN-PallaviNeural", "en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural"}
LANG_NAMES = {"ta": "tamil", "en": "english", "hi": "hindi"}


def clean_for_speech(text: str) -> str:
    text = re.sub(r"\[MOCK\]\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>•✓📎📅🎧🎉]", "", text)
    return text.strip()


def make_voice(text: str, lang: str) -> bytes:
    async def _run():
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.mp3")
            await edge_tts.Communicate(clean_for_speech(text), VOICES[lang], rate="-5%").save(path)
            with open(path, "rb") as f:
                return f.read()
    return asyncio.run(_run())


def voice_pack(d: dict, prefix: str, task_id=None):
    """d = {'ta':..., 'en':..., 'hi':...} -> 3 mp3 files (Tamil, English, Hindi).
    With task_id, the latest voices are also kept so the website can play them."""
    pack = [(f"{prefix}_{i}_{LANG_NAMES[l]}.mp3", make_voice(d[l], l))
            for i, l in enumerate(("ta", "en", "hi"), 1)]
    if task_id:
        folder = os.path.join("voice_store", str(task_id))
        os.makedirs(folder, exist_ok=True)
        for l, (_, data) in zip(("ta", "en", "hi"), pack):
            with open(os.path.join(folder, f"{l}.mp3"), "wb") as f:
                f.write(data)
    return pack


# ---------- Email ----------
def send_mail(to: str, subject: str, body: str, attachments=None):
    attachments = attachments or []
    if not SMTP_USER:  # dev mode: save voice files locally + print mail
        folder = os.path.join("audio_out", datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4])
        if attachments:
            os.makedirs(folder, exist_ok=True)
            for name, data in attachments:
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(data)
        print(f"\n--- MAIL to {to} ---\n{subject}\n{body}")
        if attachments:
            print(f"🎧 Voice files saved in: {os.path.abspath(folder)}\n")
        return
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = MAIL_FROM, to, subject
    msg.set_content(body)
    for name, data in attachments:
        msg.add_attachment(data, maintype="audio", subtype="mpeg", filename=name)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


def link(t) -> str:
    return f"{BASE_URL}/submit/{t['token']}"


# ---------- API ----------
class NewTask(BaseModel):
    intern_name: str
    intern_email: str
    title: str
    description: str
    expected_output: str
    review_days: int = 2  # days between reviews


@app.post("/tasks")
def create_task(n: NewTask):
    review = (datetime.now() + timedelta(days=n.review_days)).strftime(FMT)
    with db() as con:
        cur = con.execute(
            "INSERT INTO tasks(token,intern_name,intern_email,title,description,expected_output,review_days,review_date)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, n.intern_name, n.intern_email, n.title, n.description,
             n.expected_output, n.review_days, review))
        t = con.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone()
    exp = explain_task(t)
    send_mail(t["intern_email"], f"New task assigned: {t['title']}",
              f"Hi {t['intern_name']},\n\n🎧 Please listen to the 3 attached voice messages "
              f"(Tamil / English / Hindi) to understand your task.\n\n"
              f"📎 Submit here: {link(t)}\n📅 Review date: {t['review_date']}\n",
              attachments=voice_pack(exp, "task", t["id"]))
    return {"task_id": t["id"], "submit_link": link(t), "review_date": t["review_date"]}


HERE = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/tasks")
def list_tasks():
    with db() as con:
        rows = con.execute("SELECT id,token,intern_name,intern_email,title,status,iteration,review_date "
                           "FROM tasks ORDER BY id DESC").fetchall()
    return [dict(r, link=f"{BASE_URL}/submit/{r['token']}") for r in rows]


@app.get("/voice/{token}/{lang}")
def voice(token: str, lang: str):
    if lang not in VOICES:
        raise HTTPException(404)
    with db() as con:
        t = con.execute("SELECT id FROM tasks WHERE token=?", (token,)).fetchone()
    path = os.path.join("voice_store", str(t["id"]), f"{lang}.mp3") if t else ""
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/tasks/{task_id}")
def task_status(task_id: int):
    with db() as con:
        t = con.execute("SELECT id,title,status,iteration,review_date FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not t:
        raise HTTPException(404)
    return dict(t)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Submit task</title><style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:32px auto;padding:0 16px;color:#1a1a1a;background:#fafafa}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:12px;padding:20px;margin-bottom:16px}
h2{margin:0 0 4px}.meta{color:#666;font-size:14px;margin-bottom:12px}
.v{display:flex;align-items:center;gap:10px;margin:8px 0}.v b{width:70px}.v audio{flex:1;height:36px}
textarea{width:100%;box-sizing:border-box;padding:10px;border:1px solid #ccc;border-radius:8px;font:inherit}
button{background:#2f6fed;color:#fff;border:0;border-radius:8px;padding:10px 20px;font-size:15px;cursor:pointer}
@media(prefers-color-scheme:dark){body{background:#151515;color:#eee}.card{background:#1f1f1f;border-color:#333}.meta{color:#aaa}textarea{background:#151515;color:#eee;border-color:#444}}
</style></head><body>
<div class="card"><h2>__TITLE__</h2><div class="meta">Attempt #__IT__ &middot; Review date: __RD__</div>
<div class="v"><b>தமிழ்</b><audio controls preload="none" src="/voice/__TOKEN__/ta"></audio></div>
<div class="v"><b>English</b><audio controls preload="none" src="/voice/__TOKEN__/en"></audio></div>
<div class="v"><b>हिन्दी</b><audio controls preload="none" src="/voice/__TOKEN__/hi"></audio></div></div>
<form class="card" method="post" enctype="multipart/form-data">
<h3 style="margin-top:0">Submit your work</h3>
<p><textarea name="text" rows="9" placeholder="Paste your work or link here"></textarea></p>
<p>Or upload a file (text / code / csv / md): <input type="file" name="file"></p>
<button type="submit">Submit</button></form></body></html>"""


@app.get("/submit/{token}", response_class=HTMLResponse)
def submit_page(token: str):
    with db() as con:
        t = con.execute("SELECT * FROM tasks WHERE token=?", (token,)).fetchone()
    if not t:
        raise HTTPException(404)
    if t["status"] == "completed":
        return "<h3>🎉 This task is already completed. Great job!</h3>"
    return (PAGE.replace("__TITLE__", t["title"]).replace("__IT__", str(t["iteration"]))
            .replace("__RD__", t["review_date"]).replace("__TOKEN__", token))


@app.post("/submit/{token}", response_class=HTMLResponse)
async def submit(token: str, text: str = Form(""), file: UploadFile = File(None)):
    with db() as con:
        t = con.execute("SELECT * FROM tasks WHERE token=?", (token,)).fetchone()
        if not t or t["status"] == "completed":
            raise HTTPException(404)
        content = text
        if file and file.filename:
            content += "\n" + (await file.read()).decode("utf-8", errors="replace")
        if not content.strip():
            raise HTTPException(400, "Empty submission")
        con.execute("INSERT INTO submissions(task_id,iteration,content,created) VALUES(?,?,?,?)",
                    (t["id"], t["iteration"], content, datetime.now().strftime(FMT)))
    return f"<h3>✅ Submitted! We'll review it on {t['review_date']} and mail you.</h3>"


# ---------- Review loop ----------
def review_due():
    now = datetime.now().strftime(FMT)
    with db() as con:
        due = con.execute("SELECT * FROM tasks WHERE status='active' AND review_date<=?", (now,)).fetchall()
    for t in due:
        try:
            review_one(t)
        except Exception as e:  # never let one failure stop the others
            print("review error", t["id"], e)


def review_one(t):
    with db() as con:
        sub = con.execute("SELECT * FROM submissions WHERE task_id=? AND iteration=? ORDER BY id DESC LIMIT 1",
                          (t["id"], t["iteration"])).fetchone()
    next_review = (datetime.now() + timedelta(days=t["review_days"])).strftime(FMT)

    if not sub:  # nothing submitted -> voice reminder, push the date
        rem = {"ta": f"வணக்கம் {t['intern_name']}, உங்கள் submission இன்னும் வரலை. இந்த mail-ல இருக்க link-ல submit பண்ணுங்க. அடுத்த review தேதியும் mail-ல இருக்கு.",
               "en": f"Hi {t['intern_name']}, we have not received your submission yet. Please submit using the link in this email. The next review date is also in the email.",
               "hi": f"नमस्ते {t['intern_name']}, हमें आपका सबमिशन अभी तक नहीं मिला है। कृपया इस ईमेल के लिंक से सबमिट करें। अगली रिव्यू तारीख भी ईमेल में है।"}
        send_mail(t["intern_email"], f"Reminder: no submission yet - {t['title']}",
                  f"🎧 Listen to the voice reminders.\n\n📎 Submit: {link(t)}\n📅 New review date: {next_review}",
                  attachments=voice_pack(rem, "reminder", t["id"]))
        with db() as con:
            con.execute("UPDATE tasks SET review_date=? WHERE id=?", (next_review, t["id"]))
        return

    r = evaluate(t, sub["content"])
    fb = {"en": r["feedback_en"], "ta": r["feedback_ta"], "hi": r["feedback_hi"]}
    voices = voice_pack(fb, "feedback", t["id"])

    if r["match"]:
        with db() as con:
            con.execute("UPDATE tasks SET status='completed' WHERE id=?", (t["id"],))
        send_mail(t["intern_email"], f"🎉 Task completed: {t['title']}",
                  "🎧 Listen to the voice messages for your result.\n\nTask closed. Well done!",
                  attachments=voices)
        return

    with db() as con:
        con.execute("UPDATE tasks SET iteration=iteration+1, review_date=? WHERE id=?", (next_review, t["id"]))
    send_mail(t["intern_email"], f"Feedback (attempt #{t['iteration']}): {t['title']}",
              f"🎧 Listen to the 3 voice messages (Tamil / English / Hindi) for your feedback.\n\n"
              f"📎 Re-submit here: {link(t)}\n📅 Next review date: {next_review}",
              attachments=voices)


scheduler = BackgroundScheduler()
scheduler.add_job(review_due, "interval", minutes=CHECK_MINUTES)
scheduler.start()
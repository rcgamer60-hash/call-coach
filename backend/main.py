import os
import json
import random
import time
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from twilio.twiml.voice_response import Connect, Gather, VoiceResponse

load_dotenv(dotenv_path="../.env")

MODEL = "claude-opus-4-6"
MODEL_FAST = "claude-haiku-4-5-20251001"

REAL_CALL_RULES = (
    "CRITICAL RULES — this is a live phone call being converted to speech:\n"
    "- Talk like a real person. Use filler words: uh, yeah, right, I mean, look, so.\n"
    "- NEVER use bullet points, dashes, ellipses, or any special characters. Plain words only.\n"
    "- Keep every response to 1-2 sentences MAX. Phone calls are fast back-and-forth.\n"
    "- React to what they said before responding. Show you heard them.\n"
    "- Use contractions always: I'm, we're, don't, can't, what's.\n"
    "- NEVER say Certainly, Absolutely, Great question, I understand, Of course — robot phrases.\n"
    "- Write exactly how you would say it out loud. No punctuation except periods and question marks.\n"
)

PERSONAS = {
    "gatekeeper": {
        "name": "Linda",
        "title": "Office Receptionist",
        "difficulty": "Medium",
        "description": "Protective gatekeeper. Her job is to screen calls and keep salespeople away from the boss.",
        "system": (
            "You are Linda, receptionist at Hartwell Manufacturing. "
            "You've been screening calls for 8 years. You're friendly but you protect your boss's time fiercely. "
            "You need to know: who they are, what company, and why it can't be an email. "
            "If they're vague, you push back — politely but firmly. "
            "If they give you a real compelling reason, you'll transfer them. "
            + REAL_CALL_RULES
        ),
    },
    "skeptic": {
        "name": "Marcus",
        "title": "VP of Sales",
        "difficulty": "Hard",
        "description": "Seasoned VP who's heard every pitch. Tough, impatient, but fair — impress him and he listens.",
        "system": (
            "You are Marcus, VP of Sales at a 200-person B2B software company. "
            "You've been in sales 20 years. You've heard every pitch 50 times. "
            "You're impatient, skeptical, and you cut people off when they ramble. "
            "You demand specifics — ROI numbers, real customer names, concrete timelines. "
            "Vague answers get a hard 'look, I gotta go'. A sharp specific pitch gets your attention. "
            + REAL_CALL_RULES
        ),
    },
    "price_shopper": {
        "name": "Sarah",
        "title": "Procurement Manager",
        "difficulty": "Medium",
        "description": "Interested but obsessed with price. Every answer circles back to cost and competition.",
        "system": (
            "You are Sarah, procurement manager at a regional retail chain. "
            "You're interested but your job is to squeeze every vendor on price. "
            "You bring up competitors constantly. You hint you're getting a better deal elsewhere. "
            "You push for discounts, ask about contracts, question what's included. "
            "You never show too much interest — that's negotiating 101. "
            + REAL_CALL_RULES
        ),
    },
    "busy_exec": {
        "name": "David",
        "title": "CEO",
        "difficulty": "Very Hard",
        "description": "C-suite exec. Gives you 30 seconds. Say the wrong thing and he's gone.",
        "system": (
            "You are David, CEO of a 500-person SaaS company. "
            "You're between meetings. You gave yourself 30 seconds to decide if this call is worth your time. "
            "If they don't hook you in the first two sentences, you say 'I gotta jump, send me an email' and you're done. "
            "If they do hook you, you ask sharp questions — ROI, timeline, who else uses it. "
            "You speak in clipped, efficient sentences. You're not rude, just brutally busy. "
            + REAL_CALL_RULES
        ),
    },
    "warm_lead": {
        "name": "Jennifer",
        "title": "Marketing Director",
        "difficulty": "Easy",
        "description": "Already knows your product exists. Needs a nudge to commit — address her concerns.",
        "system": (
            "You are Jennifer, Marketing Director at a mid-size e-commerce brand. "
            "You've seen their website and you're genuinely curious. "
            "But you have real concerns: workflow disruption, onboarding time, whether support is actually good. "
            "You ask thoughtful follow-up questions. You're close to yes — you just need confidence on the details. "
            + REAL_CALL_RULES
        ),
    },
    "referral": {
        "name": "Keisha",
        "title": "Operations Manager",
        "difficulty": "Easy",
        "description": "Was referred by a mutual contact. Trusts you going in but has high expectations — don't blow it.",
        "system": (
            "You are Keisha, Operations Manager at a logistics company. "
            "Your colleague Marcus told you to take this call — said they're legit. "
            "You start warm and open because you trust Marcus's word. "
            "But your expectations are high because of that trust. "
            "If the pitch feels generic or doesn't match what Marcus described, you cool off fast. "
            "If it matches, you move fast — you don't waste time once you trust someone. "
            + REAL_CALL_RULES
        ),
    },
    "inbound": {
        "name": "Ryan",
        "title": "Head of Growth",
        "difficulty": "Medium",
        "description": "Filled out a form — he's interested but also talking to 3 competitors. Make him choose you.",
        "system": (
            "You are Ryan, Head of Growth at a 50-person startup. "
            "You filled out their form because you're actively evaluating options right now. "
            "You're also talking to two competitors this week. "
            "You ask sharp comparison questions. You want to know what makes them different, not just what they do. "
            "You have a decision deadline and you'll move fast for the right fit. "
            + REAL_CALL_RULES
        ),
    },
}


_client = None

def ai_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")


@app.get("/ping")
async def ping():
    return {"ok": True}


@app.get("/api/personas")
async def get_personas():
    return {
        k: {
            "name": v["name"],
            "title": v["title"],
            "difficulty": v["difficulty"],
            "description": v["description"],
        }
        for k, v in PERSONAS.items()
    }


class RespondRequest(BaseModel):
    persona: str
    history: list[dict]
    message: str
    product_context: Optional[str] = None


@app.post("/api/respond")
async def respond(req: RespondRequest):
    persona = PERSONAS.get(req.persona)
    if not persona:
        raise HTTPException(400, f"Unknown persona: {req.persona}")

    # Inject what the salesperson is selling so every response is contextual
    system = persona["system"]
    if req.product_context:
        system += (
            f"\n\nThe salesperson is selling: {req.product_context}. "
            "Make all your responses specific to that industry and product — "
            "use realistic objections, questions, and references that fit that context. "
            "Every call should feel fresh and different."
        )

    is_greeting = req.message == "__GREETING__"
    greeting_prompt = (
        f"(The call just connected. You are {persona['name']}, {persona['title']}. "
        "Answer the phone naturally — one short sentence, just like a real call.)"
    )
    user_msg = greeting_prompt if is_greeting else req.message
    messages = req.history + [{"role": "user", "content": user_msg}]

    client = ai_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=system,
        messages=messages,
        temperature=1,  # vary each conversation
    )
    reply = response.content[0].text.strip()
    return {"reply": reply, "persona_name": persona["name"]}


class FeedbackRequest(BaseModel):
    persona: str
    history: list[dict]
    outcome: Optional[str] = None


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    persona = PERSONAS.get(req.persona)
    if not persona:
        raise HTTPException(400, f"Unknown persona: {req.persona}")

    transcript = "\n".join(
        f"{'YOU' if m['role'] == 'user' else persona['name'].upper()}: {m['content']}"
        for m in req.history
    )

    system = (
        "You are an expert sales coach. Analyze this sales call transcript and give honest, specific feedback. "
        "Structure your response as JSON with these keys:\n"
        '- "score": number 1-10\n'
        '- "verdict": one sentence summary\n'
        '- "strengths": array of 2-3 specific things done well\n'
        '- "improvements": array of 2-3 specific things to improve\n'
        '- "tip": one actionable tip to try next time\n'
        "Be direct and specific — reference actual lines from the transcript. No fluff."
    )

    prompt = (
        f"Prospect type: {persona['name']} ({persona['title']}) — {persona['description']}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Give coaching feedback on how this pitch went."
    )

    client = ai_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception:
        data = {"score": None, "verdict": raw, "strengths": [], "improvements": [], "tip": ""}

    return data


# ---------------------------------------------------------------------------
# Twilio phone call backend
# ---------------------------------------------------------------------------

call_sessions: dict[str, dict] = {}

PERSONA_VOICE = {
    "gatekeeper": "Polly.Ruth-Generative",
    "skeptic": "Polly.Matthew-Generative",
    "busy_exec": "Polly.Matthew-Generative",
    "warm_lead": "Polly.Ruth-Generative",
    "price_shopper": "Polly.Ruth-Generative",
    "inbound": "Polly.Matthew-Generative",
    "referral": "Polly.Ruth-Generative",
}


def _twiml(text: str, voice: str, action: str) -> str:
    vr = VoiceResponse()
    gather = Gather(input="speech", action=action, method="POST", speech_timeout="1", language="en-US")
    gather.say(text, voice=voice)
    vr.append(gather)
    vr.redirect(action, method="POST")
    return str(vr)


def _xml(content: str):
    return Response(content=content, media_type="application/xml")


def _build_system(persona: dict, product: str) -> str:
    return persona["system"] + (
        f"\n\nThe salesperson is selling: {product}. "
        "Make all your responses specific to that industry and product. "
        "Every call should feel fresh and different."
    )


@app.post("/call/incoming")
async def call_incoming(CallSid: str = Form(...)):
    call_sessions[CallSid] = {"history": [], "persona": None, "product": None}
    vr = VoiceResponse()
    gather = Gather(input="speech", action="/call/got-product", method="POST", speech_timeout="auto", language="en-US")
    gather.say("Call Coach. Tell me what you're selling and I'll connect you.", voice="Polly.Ruth-Generative")
    vr.append(gather)
    vr.redirect("/call/incoming", method="POST")
    return _xml(str(vr))


@app.post("/call/got-product")
async def call_got_product(CallSid: str = Form(...), SpeechResult: str = Form(default="")):
    session = call_sessions.get(CallSid)
    if not session:
        vr = VoiceResponse()
        vr.say("Session expired. Please call back.")
        vr.hangup()
        return _xml(str(vr))

    session["product"] = SpeechResult.strip() or "general sales"
    persona_key = random.choice(list(PERSONAS.keys()))
    session["persona"] = persona_key
    persona = PERSONAS[persona_key]
    voice = PERSONA_VOICE[persona_key]
    system = _build_system(persona, session["product"])

    greeting_prompt = (
        f"(The call just connected. You are {persona['name']}, {persona['title']}. "
        "Answer the phone naturally — one short sentence.)"
    )

    client = ai_client()
    resp = client.messages.create(
        model=MODEL_FAST, max_tokens=50, system=system,
        messages=[{"role": "user", "content": greeting_prompt}], temperature=1,
    )
    greeting = resp.content[0].text.strip()
    session["history"] = [
        {"role": "user", "content": greeting_prompt},
        {"role": "assistant", "content": greeting},
    ]
    return _xml(_twiml(greeting, voice, "/call/respond"))


@app.post("/call/respond")
async def call_respond(CallSid: str = Form(...), SpeechResult: str = Form(default="")):
    session = call_sessions.get(CallSid)
    if not session or not session.get("persona"):
        vr = VoiceResponse()
        vr.say("Session expired.")
        vr.hangup()
        return _xml(str(vr))

    user_speech = SpeechResult.strip()
    if not user_speech:
        vr = VoiceResponse()
        gather = Gather(input="speech", action="/call/respond", method="POST", speech_timeout="1")
        vr.append(gather)
        vr.redirect("/call/respond", method="POST")
        return _xml(str(vr))

    persona_key = session["persona"]
    persona = PERSONAS[persona_key]
    voice = PERSONA_VOICE[persona_key]
    system = _build_system(persona, session["product"])
    session["history"].append({"role": "user", "content": user_speech})

    client = ai_client()
    resp = client.messages.create(
        model=MODEL_FAST, max_tokens=75, system=system,
        messages=session["history"], temperature=1,
    )
    reply = resp.content[0].text.strip()
    session["history"].append({"role": "assistant", "content": reply})
    return _xml(_twiml(reply, voice, "/call/respond"))


@app.post("/call/status")
async def call_status(CallSid: str = Form(...), CallStatus: str = Form(default="")):
    if CallStatus in ("completed", "failed", "busy", "no-answer", "canceled"):
        call_sessions.pop(CallSid, None)
    return {"ok": True}

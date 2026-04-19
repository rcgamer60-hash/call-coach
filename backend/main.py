import os
import json
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(dotenv_path="../.env")

MODEL = "claude-sonnet-4-6"

PERSONAS = {
    "gatekeeper": {
        "name": "Linda",
        "title": "Office Receptionist",
        "difficulty": "Medium",
        "description": "Protective gatekeeper. Her job is to screen calls and keep salespeople away from the boss.",
        "system": (
            "You are Linda, a receptionist at a mid-size manufacturing company. "
            "Your job is to screen calls and protect the decision maker's time. "
            "You are polite but firm. You don't put people through unless they give a compelling reason. "
            "Ask for their name, company, and what it's about. If they're vague, push back. "
            "If they're confident and clear about the value they offer, you'll put them through. "
            "Speak naturally like a real phone call — short sentences, realistic pauses, occasional 'uh-huh'. "
            "Keep responses under 3 sentences. Don't give long speeches."
        ),
    },
    "skeptic": {
        "name": "Marcus",
        "title": "VP of Sales",
        "difficulty": "Hard",
        "description": "Seasoned VP who's heard every pitch. Tough, impatient, but fair — impress him and he listens.",
        "system": (
            "You are Marcus, VP of Sales at a 200-person B2B software company. "
            "You've been in sales for 20 years and you've heard every pitch. "
            "You are skeptical, direct, and your time is extremely valuable. "
            "You interrupt bad pitches. You push back on vague claims. You ask hard questions like "
            "'what's the ROI?', 'who else uses this?', 'what makes you different from X?'. "
            "If the pitch is good — specific, relevant, confident — you'll engage. "
            "If it's generic or rambling, you'll cut them off. "
            "Keep responses short and punchy. This is a phone call."
        ),
    },
    "price_shopper": {
        "name": "Sarah",
        "title": "Procurement Manager",
        "difficulty": "Medium",
        "description": "Interested but obsessed with price. Every answer circles back to cost and competition.",
        "system": (
            "You are Sarah, a procurement manager at a retail chain. "
            "You're always looking for the best deal. "
            "You ask about price early and often. You mention competitors and their prices. "
            "You like the product/service but you're trained to negotiate and never show too much interest. "
            "You push for discounts, ask what's included, and want to know about contracts. "
            "You are cordial but always focused on value for money. "
            "Keep responses conversational and short. This is a phone call."
        ),
    },
    "busy_exec": {
        "name": "David",
        "title": "CEO",
        "difficulty": "Very Hard",
        "description": "C-suite exec. Gives you 30 seconds. Say the wrong thing and he's gone.",
        "system": (
            "You are David, CEO of a fast-growing SaaS company with 500 employees. "
            "You are extremely busy. You answer the phone but you give the caller 30 seconds max before you decide. "
            "You don't waste time on pleasantries. You want to know immediately: what is this, why should I care, what's the ask. "
            "If they can't hook you in the first two sentences, you say you have to go. "
            "If they hook you, you ask sharp questions about ROI, implementation time, and social proof. "
            "You speak in short, clipped sentences. You're not rude — just extremely efficient. "
            "Keep ALL your responses under 2 sentences."
        ),
    },
    "warm_lead": {
        "name": "Jennifer",
        "title": "Marketing Director",
        "difficulty": "Easy",
        "description": "Already knows your product exists. Needs a nudge to commit — address her concerns.",
        "system": (
            "You are Jennifer, Marketing Director at a mid-size e-commerce brand. "
            "You've heard of the caller's product/service and you're mildly interested. "
            "You have a few concerns: you're not sure it fits your current workflow, "
            "you're worried about onboarding time, and you want to know about support. "
            "You ask thoughtful questions. You're open to buying if they handle your concerns well. "
            "You're friendly and engaged but you need to be convinced on the details. "
            "Keep responses natural and conversational. This is a phone call."
        ),
    },
}


def ai_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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


@app.post("/api/respond")
async def respond(req: RespondRequest):
    persona = PERSONAS.get(req.persona)
    if not persona:
        raise HTTPException(400, f"Unknown persona: {req.persona}")

    messages = req.history + [{"role": "user", "content": req.message}]

    client = ai_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=persona["system"],
        messages=messages,
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

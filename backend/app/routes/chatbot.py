import os
import requests
from fastapi import APIRouter, Depends
from app.utils.auth_utils import get_current_user
from app.models.user_model import User
from pydantic import BaseModel

router = APIRouter()

# -----------------------------
# MEMORY STORE
# -----------------------------
REPORT_MEMORY = {}

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str
    module: str   # career / mental_health / burnout

# -----------------------------
# STORE REPORT
# -----------------------------
def store_report(user_id, report, module, raw_data=None):
    if user_id not in REPORT_MEMORY:
        REPORT_MEMORY[user_id] = {}
    
    REPORT_MEMORY[user_id][module] = {
        "report": report,
        "data": raw_data
    }

# -----------------------------
# CHATBOT API
# -----------------------------
@router.post("/chat")
def chatbot(request: ChatRequest, current_user: User = Depends(get_current_user)):

    user_id = current_user.email
    question = request.question
    module = request.module

    if user_id not in REPORT_MEMORY or module not in REPORT_MEMORY[user_id]:
        return {
            "response": f"No {module} report found. Please generate it first."
        }

    context = REPORT_MEMORY[user_id][module]
    report_text = context["report"]
    raw_data = context.get("data", "No raw data available.")
    
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if module == "career":
        system_role = "You are an Expert Career Strategist. Use the user's specific data (CGPA, Course, interest) to give precise, non-generic advice."
    elif module == "mental_health":
        system_role = "You are a Compassionate Mental Health Consultant. Use the user's wellness metrics (Sleep, Stress, Mood) to provide supportive, data-driven feedback."
    else:
        system_role = "You are a Workplace Wellness Coach. Use the user's work-life metrics (Hours, Fatigue, Balance) to suggest actionable recovery steps."

    prompt = f"""
OBJECTIVE:
You are a highly capable AI assistant specialized in {module}. Your goal is to help the user interpret and act upon their {module} analysis.

USER PROFILE (RAW DATA):
{raw_data}

ANALYSIS REPORT:
{report_text}

GUIDELINES:
1. **Be Data-Driven**: Reference the user's specific metrics (e.g., their CGPA, Stress level, or Work hours) in your answers.
2. **Be Specific**: Avoid generic advice. Instead of saying "improve your skills," say "since you are interested in {raw_data.get('Interest', 'your field') if isinstance(raw_data, dict) else 'your field'}, you should focus on..."
3. **Be Conversational but Professional**: Treat the user with respect and empathy.
4. **Contextual Awareness**: If the user asks a follow-up, relate it back to their profile.
5. **No Medical Diagnosis**: If in the mental health/burnout module, never give a formal diagnosis.

User Question:
{question}
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "CareerSphereAI"
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500
            },
            timeout=30
        )

        result = response.json()

        if "choices" in result:
            return {
                "response": result["choices"][0]["message"]["content"]
            }

        return {"response": f"AI Error: {str(result)}"}

    except Exception as e:
        return {"response": f"System Error: {str(e)}"}
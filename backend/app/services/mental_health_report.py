import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def generate_mental_health_report(data, prediction, confidence):

    prompt = f"""
ROLE:
You are a Compassionate and Supportive Mental Health Consultant. Your goal is to provide a safe, non-judgmental, and highly personalized wellness analysis.

CONTEXT:
Analyze the following lifestyle and behavioral data to generate a "Holistic Well-being & Resilience Report". Address the user's current mental state ({prediction}) with empathy and practical guidance.

---------------------------------------------------------
USER WELLNESS DATA:
---------------------------------------------------------
- Stress Level: {data['Stress_Level']}
- Current Mood: {data['Mood']}
- Anxiety Level: {data['Anxiety_Level']}
- Sleep Quality: {data['Sleep_Quality']}
- Social Interaction: {data['Social_Interaction']}
- Screen Time: {data['Screen_Time_Hours']} hours/day
- Productivity: {data['Productivity_Level']}
- Routine Consistency: {data['Routine_Consistency']}
- Physical Activity: {data['Exercise_Frequency']}
- Diet Quality: {data['Diet_Quality']}
- Current Workload: {data['Workload_Level']}

MODEL PREDICTION:
- Mental Health Status: {prediction}
- Model Confidence: {confidence}%

---------------------------------------------------------
REPORT STRUCTURE (Markdown):
---------------------------------------------------------
1. ### Well-being Snapshot
   - A soothing 2-3 sentence summary of their current wellness status.

2. ### Behavioral & Lifestyle Analysis
   - Analyze how factors like Screen Time ({data['Screen_Time_Hours']}h) and Sleep ({data['Sleep_Quality']}) are interacting with their {data['Stress_Level']} stress level.
   - Contrast their {data['Exercise_Frequency']} exercise with their {data['Productivity_Level']} productivity.

3. ### Key Resilience Factors & Risks
   - Identify what is working well (e.g., strong routine) and what needs immediate attention.

4. ### Personalized "Micro-Habit" Plan
   - 3-5 small, actionable steps they can take *today*.

5. ### Stress Management & Grounding Techniques
   - Provide 2 specific techniques tailored to an anxiety level of {data['Anxiety_Level']}.

6. ### Daily Harmony Routine
   - A simple morning/evening routine suggestion based on {data['Routine_Consistency']}.

7. ### Supportive Closing & Resources
   - A final encouraging word.

---------------------------------------------------------
CRITICAL GUIDELINES:
---------------------------------------------------------
- **DISCLAIMER**: Always start or end with: "I am an AI, not a doctor. This report is for wellness purposes and does not constitute a medical diagnosis."
- **TONE**: Empathetic, calm, safe, and empowering.
- **SAFETY**: If signs of severe distress are present (though not explicitly in this data, keep it professional), always suggest seeking professional help.
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
                    {"role": "system", "content": "You are a safe mental health assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000
            },
            timeout=30
        )

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        return f"Report generation failed: {result}"

    except Exception as e:
        return f"Error: {str(e)}"
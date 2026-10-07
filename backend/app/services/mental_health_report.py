import os
from app.services.openrouter_service import query_openrouter

def generate_fallback_mental_health_report(data: dict, prediction: str, confidence: float) -> str:
    stress = data.get('Stress_Level', 'Medium')
    mood = data.get('Mood', 'Neutral')
    anxiety = data.get('Anxiety_Level', 'Medium')
    sleep = data.get('Sleep_Quality', 'Average')
    screen_time = data.get('Screen_Time_Hours', 5)
    productivity = data.get('Productivity_Level', 'Medium')
    routine = data.get('Routine_Consistency', 'Moderate')
    exercise = data.get('Exercise_Frequency', 'Occasional')
    diet = data.get('Diet_Quality', 'Average')
    workload = data.get('Workload_Level', 'Medium')

    return f"""### 🧘 Well-being Snapshot
Your overall **Mental Health Well-being Assessment** indicates a status of **{prediction}** with a model confidence of **{confidence}%**. 

*Disclaimer: I am an AI assistant, not a medical professional. This analysis provides personal lifestyle recommendations and non-medical wellness insights.*

Currently reporting a **{mood}** mood alongside **{stress}** stress and **{anxiety}** anxiety levels, your system is navigating a workload level rated as **{workload}**.

---

### 🔍 Behavioral & Lifestyle Analysis
- **Screen Time & Cognitive Load**: Logging **{screen_time} hours/day** of screen time directly impacts eye strain, sleep architecture, and mental fatigue.
- **Sleep Quality Impact**: Your sleep quality is rated as **{sleep}**. Restorative sleep is the foundational pillar for emotional regulation and cognitive clarity.
- **Physical Activity & Diet**: Combining **{exercise}** exercise with a **{diet}** diet influences mood stability and natural dopamine baseline.
- **Productivity & Routine**: Operating with **{routine}** routine consistency and **{productivity}** productivity shows active resilience under current demands.

---

### 🛡️ Key Resilience Factors & Risks
- **Current Strengths**: Maintaining structured effort despite **{workload}** workload.
- **Primary Risk Factors**: Extended screen engagement ({screen_time}h) paired with **{stress}** stress levels can exacerbate fatigue over time.

---

### 🌱 Personalized "Micro-Habit" Plan
1. **Screen-Free Sunset**: Shut down all bright display screens 45 minutes prior to sleep to boost melatonin production.
2. **20-20-20 Rule**: Every 20 minutes of screen usage, focus on an object 20 feet away for 20 seconds.
3. **Hydration & Movement Break**: Take a 5-minute walk and drink water every 2 hours of seated work.
4. **Daily Mind Dump**: Write down all urgent tasks before bedtime to clear mental overhead.

---

### 🧘 Stress Management & Grounding Techniques
- **Box Breathing (4-4-4-4)**: Inhale for 4 seconds, hold for 4, exhale for 4, hold for 4. Repeat for 3 cycles during high stress spikes.
- **5-4-3-2-1 Sensory Grounding**: Identify 5 things you see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste when feeling anxious.

---

### 🌅 Daily Harmony Routine
- **Morning**: 10 minutes of morning sunlight exposure + light stretching before checking phone notifications.
- **Evening**: Gentle digital detox + warm chamomile tea or calm listening to restore peaceful sleep.

> *"Prioritizing your mental peace is not a luxury; it is the fundamental engine of your success and health."*
"""

def generate_mental_health_report(data: dict, prediction: str, confidence: float) -> str:
    system_role = "You are a safe, compassionate mental health and wellness consultant."
    prompt = f"""
USER WELLNESS DATA:
- Stress Level: {data.get('Stress_Level')}
- Current Mood: {data.get('Mood')}
- Anxiety Level: {data.get('Anxiety_Level')}
- Sleep Quality: {data.get('Sleep_Quality')}
- Social Interaction: {data.get('Social_Interaction')}
- Screen Time: {data.get('Screen_Time_Hours')} hours/day
- Productivity: {data.get('Productivity_Level')}
- Routine: {data.get('Routine_Consistency')}
- Exercise: {data.get('Exercise_Frequency')}
- Workload: {data.get('Workload_Level')}

MODEL PREDICTION:
- Mental Health Status: {prediction}
- Confidence: {confidence}%

Generate a soothing, structured, empathetic well-being report in Markdown format with disclaimer, lifestyle analysis, micro-habits, and grounding techniques. Limit to 650 words.
"""
    ai_content = query_openrouter(system_role, prompt, max_tokens=1500, timeout=6)
    if ai_content:
        return ai_content

    return generate_fallback_mental_health_report(data, prediction, confidence)
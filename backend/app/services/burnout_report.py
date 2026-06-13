import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def generate_burnout_report(data, prediction, confidence):

    prompt = f"""
ROLE:
You are a Workplace Wellness & Performance Coach specializing in occupational burnout and sustainable productivity.

CONTEXT:
Analyze the following workload and lifestyle datasets to generate a "Sustainable Performance & Burnout Prevention Report". Your analysis should be clinical yet actionable, focusing on the user's current risk level: {prediction}.

---------------------------------------------------------
USER WORK-LIFE DATA:
---------------------------------------------------------
- Work Hours: {data['Work_Hours_Per_Day']} hours/day
- Workload Level: {data['Workload_Level']}
- Deadline Pressure: {data['Deadline_Pressure']}
- Stress Level: {data['Stress_Level']}
- Fatigue Level: {data['Fatigue_Level']}
- Sleep: {data['Sleep_Hours']} hours/day
- Exercise: {data['Exercise_Frequency']}
- Work-Life Balance Rating: {data['Work_Life_Balance']}

MODEL PREDICTION:
- Burnout Risk Level: {prediction}
- Model Confidence: {confidence}%

---------------------------------------------------------
REPORT STRUCTURE (Markdown):
---------------------------------------------------------
1. ### Burnout Risk Assessment
   - A concise summary of the current risk based on the {prediction} status and {data['Fatigue_Level']} fatigue.

2. ### Critical Pressure Points
   - Identify which factors (e.g., {data['Deadline_Pressure']} pressure vs {data['Work_Hours_Per_Day']} hours) are the primary drivers of stress.
   - Mention the impact of {data['Sleep_Hours']} hours of sleep on their {data['Fatigue_Level']} fatigue.

3. ### Work-Life Balance Audit
   - Evaluate the current {data['Work_Life_Balance']} rating against their actual work hours and exercise frequency.

4. ### Immediate "Pressure Release" Actions
   - Provide 3 specific actions the user can take in the next 24 hours to reduce immediate stress.

5. ### Long-Term Sustainability Strategy
   - Recommendations for workload management, boundaries, and lifestyle adjustments.

6. ### Recommended Daily Energy Management
   - A sample schedule that incorporates breaks and recovery based on their workload.

7. ### Performance Coach's Final Take
   - A final motivating but firm piece of advice on maintaining health while achieving goals.

---------------------------------------------------------
TONE:
---------------------------------------------------------
Professional, direct, performance-oriented, yet deeply concerned with long-term health. Avoid fluff; use the numbers provided to justify every recommendation.
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
                    {"role": "system", "content": "You are a burnout advisor."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000
            },
            timeout=30
        )

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        return f"Error: {result}"

    except Exception as e:
        return f"Error generating report: {str(e)}"
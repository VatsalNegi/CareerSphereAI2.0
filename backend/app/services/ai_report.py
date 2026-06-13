import os
import requests

# -----------------------------
# LOAD API KEY
# -----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -----------------------------
# AI REPORT GENERATION
# -----------------------------
def generate_ai_report(data, prediction, confidence):

    prompt = f"""
ROLE:
You are a Senior Strategic Career Consultant and Industry Expert with 15+ years of experience in global recruitment and academic advisory. 

CONTEXT:
Analyze the following student profile and generate a comprehensive, high-impact Career Strategy Report. Your advice must be data-driven, specifically addressing the student's current standing and career aspirations.

---------------------------------------------------------
STUDENT PROFILE DATA:
---------------------------------------------------------
- University: {data['University']}
- Course: {data['Course']}
- Department: {data['Department']}
- Branch: {data['Branch']}
- Semester: {data['Semester']}
- CGPA: {data['CGPA']}
- Career Interest: {data['Interest']}
- Mode of Study: {data['Mode_of_Study']}
- Institution Type: {data['Institution_Type']}
- Location: {data['Location']}
- Medium of Education: {data['Medium_of_Education']}

MODEL PREDICTION:
- Career Readiness Status: {prediction}
- Model Confidence: {confidence}%

---------------------------------------------------------
INSTRUCTIONS & STRUCTURE:
---------------------------------------------------------
Generate the report using the following structure. Use professional Markdown (### headings, bold text, bullet points).

1. ### Executive Summary
   - Provide a 2-3 sentence overview of the student's current profile and the "Career Readiness" prediction.

2. ### Academic Performance & Benchmark Analysis
   - Analyze the current CGPA ({data['CGPA']}) against industry standards for {data['Branch']}.
   - Identify if the student is in a "Critical", "Average", or "Excel" zone.
   - Mention the impact of the current semester ({data['Semester']}) on upcoming placement/internship seasons.

3. ### Core Strengths (Based on Data)
   - List 3-4 strengths derived from the profile (e.g., specific field interest, institution type, or educational medium).

4. ### Strategic Skill Gap Analysis
   - Identify specific technical and soft skills the student likely lacks based on their {data['Branch']} and interest in {data['Interest']}.
   - Compare the current profile against "Day 1 Ready" industry requirements.

5. ### Personalized 6-Month Roadmap
   - Provide a month-by-month actionable plan.
   - Include specific certifications, projects, or internship types.

6. ### Industry Outlook for {data['Interest']}
   - Provide current trends, salary expectations (junior level), and key hiring companies in {data['Location']} or globally.

7. ### Expert Career Advice
   - A final closing statement that is both realistic and motivating.

TONE: 
Professional, analytical, empathetic, and highly actionable. Limit to ~800 words. Avoid generic phrases; use the specific profile data in every section.
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
                    {"role": "system", "content": "You are an expert career advisor."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000
            },
            timeout=30
        )

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        return f"AI Report generation failed: {result}"

    except Exception as e:
        return f"AI Report error: {str(e)}"
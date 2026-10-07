import os
from app.services.openrouter_service import query_openrouter

def generate_fallback_career_report(data: dict, prediction: str, confidence: float) -> str:
    university = data.get('University', 'University')
    course = data.get('Course', 'Degree Program')
    branch = data.get('Branch', 'Specialization')
    cgpa = float(data.get('CGPA', 7.5))
    semester = data.get('Semester', 6)
    interest = data.get('Interest', 'Job')
    location = data.get('Location', 'Urban')
    
    # Benchmarking CGPA
    if cgpa >= 8.5:
        cgpa_tier = "Distinction / Top Tier (Excel Zone)"
        cgpa_eval = f"Your current CGPA of **{cgpa}** places you in the top tier academically, demonstrating strong subject mastery and consistency."
    elif cgpa >= 7.0:
        cgpa_tier = "Competitive / Standard Zone"
        cgpa_eval = f"Your CGPA of **{cgpa}** meets standard industry cut-offs (65-70%+), but raising it closer to 8.0+ will significantly expand your eligibility for top-tier campus recruitments."
    else:
        cgpa_tier = "Needs Immediate Focus (Critical Zone)"
        cgpa_eval = f"Your current CGPA of **{cgpa}** is near or below typical placement cut-offs. Priority must be placed on grade enhancement in upcoming exams."

    return f"""### 🚀 Executive Summary
Based on our machine learning assessment model, your overall **Career Readiness Status** is categorized as **{prediction}** with a model confidence rating of **{confidence}%**. 

As a **Semester {semester}** student pursuing **{course} ({branch})** at **{university}**, your academic background provides a solid foundation. Your primary goal is targeted towards **{interest}**, and this report outlines strategic actions to ensure high marketability upon graduation.

---

### 📊 Academic Performance & Benchmark Analysis
- **Current CGPA**: `{cgpa}` / 10.0
- **Academic Benchmark Tier**: `{cgpa_tier}`
- **Evaluation**: {cgpa_eval}
- **Semester Impact**: Being in **Semester {semester}**, you are entering a critical phase. Top companies begin shortlisting candidate resumes 3–6 months prior to graduation.

---

### 💪 Core Strengths (Based on Profile Data)
1. **Targeted Specialization**: Studying `{branch}` aligns closely with expanding opportunities in tech, analytics, and technical consultancy in `{location}` centers.
2. **Domain Alignment**: Clear orientation towards **{interest}** provides clarity in selecting certifications and project portfolios.
3. **Institutional Environment**: Program structure at `{university}` facilitates structured learning and academic network access.

---

### 🛠️ Strategic Skill Gap Analysis
To transition from **{prediction}** readiness to **Industry-Ready**, address these specific skill gaps:
- **Technical Competencies**: Deepen practical expertise in core domain tooling for `{branch}` (e.g., industry-standard software, data analysis tools, or modern frameworks).
- **Project Portfolio**: Build at least **2 end-to-end production-grade projects** hosted on GitHub/cloud repositories rather than simple academic assignments.
- **Aptitude & Technical Screening**: Daily practice with quantitative reasoning, algorithmic problem solving, and domain fundamentals.
- **Communication & Professional Branding**: Refine LinkedIn profile optimization, resume formatting (ATS compliant), and mock interview readiness.

---

### 📅 Personalized 6-Month Action Plan

#### Month 1–2: Foundation & Portfolio Building
- Master advanced domain concepts related to **{branch}**.
- Launch an open-source or individual showcase project matching **{interest}** requirements.
- Target technical certifications relevant to modern industry standard tools.

#### Month 3–4: Internship & Industry Exposure
- Apply for virtual/onsite summer internships or research assistantships.
- Refine ATS-optimized resume emphasizing quantifiable project outcomes.
- Practice mock interviews and live problem-solving sessions weekly.

#### Month 5–6: Final Placement & Career Launch
- Target campus placements and direct referral channels across top hiring hubs in `{location}`.
- Complete final capstone project with measurable real-world metrics.
- Participate in hackathons, industry webinars, and alumni networking sessions.

---

### 🎯 Industry Outlook for {interest}
- **Market Demand**: Technical roles in **{branch}** are seeing steady demand across digital-first enterprises and high-growth startups.
- **Key Hiring Focus**: Employers prioritize candidates with practical project deployment, problem-solving agility, and strong work ethics over pure theory.
- **Salary/Package Trends**: Junior roles typically offer competitive starter packages, with fast growth trajectories for continuous learners.

---

### 🌟 Expert Career Advice
> *"Success in the modern hiring landscape is determined by proven capability, persistent execution, and continuous upskilling. Stay focused on building real proof of competence!"*
"""

def generate_ai_report(data: dict, prediction: str, confidence: float) -> str:
    system_role = "You are a Senior Strategic Career Consultant with 15+ years of experience in global recruitment and academic advisory."
    prompt = f"""
STUDENT PROFILE:
- University: {data.get('University')}
- Course: {data.get('Course')}
- Department: {data.get('Department')}
- Branch: {data.get('Branch')}
- Semester: {data.get('Semester')}
- CGPA: {data.get('CGPA')}
- Career Interest: {data.get('Interest')}
- Location: {data.get('Location')}

MODEL PREDICTION:
- Career Readiness Status: {prediction}
- Confidence: {confidence}%

Generate a highly actionable, structured, professional Career Strategy Report in Markdown format with headings (###), bold text, bullet points, executive summary, skill gap analysis, and 6-month roadmap. Keep it within 700 words.
"""
    ai_content = query_openrouter(system_role, prompt, max_tokens=1500, timeout=6)
    if ai_content:
        return ai_content
    
    return generate_fallback_career_report(data, prediction, confidence)
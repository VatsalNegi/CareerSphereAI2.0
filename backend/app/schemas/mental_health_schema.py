from pydantic import BaseModel

class MentalHealthInput(BaseModel):
    Stress_Level: str
    Mood: str
    Anxiety_Level: str
    Sleep_Quality: str
    Social_Interaction: str
    Screen_Time_Hours: float
    Productivity_Level: str
    Routine_Consistency: str
    Exercise_Frequency: str
    Diet_Quality: str
    Workload_Level: str
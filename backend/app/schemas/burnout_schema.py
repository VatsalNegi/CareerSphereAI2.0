from pydantic import BaseModel

class BurnoutInput(BaseModel):
    Work_Hours_Per_Day: float
    Workload_Level: str
    Deadline_Pressure: str
    Stress_Level: str
    Fatigue_Level: str
    Sleep_Hours: float
    Exercise_Frequency: str
    Work_Life_Balance: str
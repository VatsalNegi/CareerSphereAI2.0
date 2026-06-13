from pydantic import BaseModel

class CareerInput(BaseModel):
    University: str
    Course: str
    Department: str
    Branch: str
    Semester: int
    CGPA: float
    Interest: str
    Mode_of_Study: str
    Institution_Type: str
    Location: str
    Medium_of_Education: str
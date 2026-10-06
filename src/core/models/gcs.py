from enum import IntEnum
from pydantic import BaseModel, Field
from typing import Optional


class EyeOpening(IntEnum):
    NONE = 1
    TO_PAIN = 2
    TO_VOICE = 3
    SPONTANEOUS = 4


class VerbalResponse(IntEnum):
    NONE = 1
    INCOMPREHENSIBLE = 2
    INAPPROPRIATE = 3
    CONFUSED = 4
    ORIENTED = 5


class MotorResponse(IntEnum):
    NONE = 1
    EXTENSION = 2
    FLEXION_ABNORMAL = 3
    WITHDRAWAL = 4
    LOCALIZING = 5
    OBEYING = 6


class GCSScore(BaseModel):
    eye_opening: EyeOpening = Field(description="Eye opening response score (1-4)")
    verbal_response: VerbalResponse = Field(description="Verbal response score (1-5)")
    motor_response: MotorResponse = Field(description="Motor response score (1-6)")

    @property
    def total_score(self) -> int:
        return int(self.eye_opening) + int(self.verbal_response) + int(self.motor_response)

    @property
    def severity(self) -> str:
        score = self.total_score
        if score <= 8:
            return "grave"
        elif score <= 12:
            return "moderato"
        else:
            return "lieve"


class GCSQuestion(BaseModel):
    id: str
    category: str
    text: str
    options: dict[int, str]
    answer: Optional[int] = None


class GCSQuestionnaire(BaseModel):
    questions: list[GCSQuestion] = Field(default_factory=list)
    problem_description: str = ""
    answers: dict[str, int] = Field(default_factory=dict)

    def calculate_score(self) -> Optional[GCSScore]:
        if not all(k in self.answers for k in ["eye", "verbal", "motor"]):
            return None
        return GCSScore(
            eye_opening=EyeOpening(self.answers["eye"]),
            verbal_response=VerbalResponse(self.answers["verbal"]),
            motor_response=MotorResponse(self.answers["motor"]),
        )

from pydantic import BaseModel


class SkillGroup(BaseModel):
    category: str
    items: list[str]

class Education(BaseModel):
    degree: str
    institution: str

class Language(BaseModel):
    name: str
    proficiency: str

class Experience(BaseModel):
    id: str
    company: str
    title: str
    startDate: str
    endDate: str | None
    highlights: list[str]

class Profile(BaseModel):
    profileId: str
    name: str
    headline: str
    summary: str
    experience: list[Experience]
    skills: list[SkillGroup]
    education: list[Education]
    languages: list[Language]

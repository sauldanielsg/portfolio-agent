export type SkillGroup = {
    category: string;
    items: string[];
};

export type Education = {
    degree: string;
    institution: string;
}

export type Language = {
    name: string;
    proficiency: string;
}

export type Experience = {
    id: string;
    company: string;
    title: string;
    startDate: string;
    endDate: string | null;
    highlights: string[];
};

export type Profile = {
    profileId: string;
    name: string;
    headline: string;
    summary: string;
    experience: Experience[];
    skills: SkillGroup[];
    education: Education[];
    languages: Language[];
};
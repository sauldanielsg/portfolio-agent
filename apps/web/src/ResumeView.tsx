import type { Profile } from "./types";

type Props = {
    profile: Profile;
}

export function ResumeView({ profile }: Props) {
    return (
        <article>
            <h1>{profile.name}</h1>
            <h2>{profile.headline}</h2>
            <p>{profile.summary}</p>

            <h2>Experience</h2>
            {profile.experience.map((experience) => (
                <section key={experience.id}>
                    <h3>
                        {experience.title} · {experience.company}
                    </h3>
                    <p>
                        {experience.startDate} – {experience.endDate ?? "Present"}
                    </p>
                    <ul>
                        {experience.highlights.map((highlight) => (
                        <li key={highlight}>{highlight}</li>
                        ))}
                    </ul>
                </section>
            ))}
            <h2>Skills</h2>
            {profile.skills.map((group) => (
            <section key={group.category}>
                <h3>{group.category}</h3>
                <p>{group.items.join(", ")}</p>
            </section>
            ))}
            {profile.education.length > 0 && <h2>Education</h2>}
            {profile.education.map((education) => (
                <section key={`${education.institution}-${education.degree}`}>
                    <h3>{education.degree}</h3>
                    <p>{education.institution}</p>
                </section>
            ))}
            {profile.languages.length > 0 && <h2>Languages</h2>}
            {profile.languages.map((language) => (
                <p key={language.name}>{language.name}: {language.proficiency}</p>
            ))}
        </article>
    )
}

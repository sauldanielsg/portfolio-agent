import type { Citation } from "../../types/chat";

type Props = {
  citations: Citation[];
};

export function SourceCitations({ citations }: Props) {
  if (citations.length === 0) return null;

  return (
    <div aria-label="Answer sources">
      <p>Sources</p>
      {citations.map((citation) => (
        <details key={citation.chunkId}>
          <summary>{citation.title}</summary>
          <blockquote>{citation.text}</blockquote>
        </details>
      ))}
    </div>
  );
}

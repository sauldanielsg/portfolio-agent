export type Citation = {
  chunkId: string;
  sourceId: string;
  documentVersion: string;
  title: string;
  text: string;
};

export type Message = {
  role: "assistant" | "user";
  content: string;
  citations?: Citation[];
};

export type ChatResponse = {
  message: string;
  action: "show_resume" | null;
  citations: Citation[];
};

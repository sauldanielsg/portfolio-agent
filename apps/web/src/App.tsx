import { useState, useEffect } from 'react'
import './App.css'
import type { Profile } from './types';
import { ResumeView } from './ResumeView';

type Message = {
  role: "assistant" | "user";
  content: string;
};

type ChatResponse = {
  message: string;
  action: "show_resume" | null;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    content: 'Hi! Ask me about my full-stack experience or request my resume.'
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResume, setShowResume] = useState(false);
  const [isChatMinimized, setIsChatMinimized] = useState(false);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [unavailableMessage, setUnavailableMessage] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadProfile() {
      try {
        const response = await fetch("/api/v1/profile", {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("Could not load the resume.");
        }

        const data: Profile = await response.json();
        setProfile(data);
      } catch (error) {
        if (!controller.signal.aborted) {
          setProfileError(
            error instanceof Error ? error.message : "Could not load the resume."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setProfileLoading(false);
        }
      }
    }

    void loadProfile();

    return () => controller.abort();
  }, []);

  async function handleSubmit(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
    ]);
    setInput("");
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch("/api/v1/chat", {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-10).map((message) => ({
            role: message.role,
            content: message.content.slice(0, 4000),
          }))
        }),
      });

      if (response.status === 503 || response.status === 429) {
        const data = await response.json();
        const detail = data.detail ?? data;
        if (detail.error === "AI_UNAVAILABLE" || detail.error === "DAILY_LIMIT_REACHED") {
          setUnavailableMessage(detail.message ?? "I'm taking a break. Please check back later.");
          return;
        }
      }

      if(!response.ok) {
        throw new Error(`Request failed: ${response.status}`)
      }

      const data: ChatResponse = await response.json();
      setMessages((previous) => [
        ...previous,
        { role: "assistant", content: data.message },
      ]);
      if (data.action === "show_resume") {
        if (profile) {
          setShowResume(true);
          setIsChatMinimized(true);
        } else {
          setError(
            profileLoading
              ? "The resume is still loading. Please try again in a moment."
              : profileError ?? "The resume is unavailable."
          );
        }
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Something went wrong."
      );
    } finally {
      setIsLoading(false);
    }
}

  if (unavailableMessage) {
    return (
      <main className="resting-screen">
        <h1>{profile?.name ?? "Saul D Silva Garcia"}</h1>
        <p>{profile?.headline ?? "Full-Stack Software Engineer"}</p>
        <h2>Taking a break</h2>
        <p role="status">{unavailableMessage}</p>
        <button type="button" onClick={() => setUnavailableMessage(null)}>
          Back to conversation
        </button>
      </main>
    );
  }

  return (
    <>
      {showResume && profile && (
        <main>
          <ResumeView profile={profile} />
        </main>
      )}

      {isChatMinimized ? (
        <button onClick={() => setIsChatMinimized(false)}>
          Continue conversation
        </button>
      ) : (
        <section aria-label="Portfolio assistant">
          <div role="log" aria-label="Conversation" aria-live="polite">
          {messages.map((message, index) => (
            <p key={index}>
              <strong>{message.role}:</strong> {message.content}
            </p>
          ))}
          </div>
          {error && <p role="alert">{error}</p>}
          <form onSubmit={handleSubmit}>
            <label htmlFor="message">Your message</label>
            <input
              id="message"
              name="message"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              maxLength={200}
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !input.trim()}>
              {isLoading ? "Thinking..." : "Send"}
            </button>
          </form>

          {showResume && (
            <button onClick={() => setIsChatMinimized(true)}>
              Minimize conversation
            </button>
          )}
        </section>
      )}
    </>
  )
}

export default App

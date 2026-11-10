import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const REFUSAL = 'I cannot answer this based on the provided eBook.'
const MAX_LEN = 1000

type Source = { source: string | null; page: number | null; score: number }

type Message =
  | { role: 'user'; text: string }
  | {
      role: 'assistant'
      text: string
      confidence: number
      grounded: boolean
      sources: Source[]
    }
  | { role: 'error'; text: string }

function errorText(status: number) {
  if (status === 400) return 'Please enter a question.'
  if (status === 422) return 'Your question is too long (max 1000 characters).'
  if (status === 503) return 'The service is temporarily unavailable. Please try again in a minute.'
  return `Something went wrong (error ${status}).`
}

function tone(score: number) {
  if (score >= 0.8)
    return { pill: 'border-accent-400/30 bg-accent-400/10 text-accent-200', bar: 'bg-accent-400' }
  if (score >= 0.5)
    return { pill: 'border-white/15 bg-white/[0.05] text-zinc-300', bar: 'bg-zinc-400' }
  return { pill: 'border-rose-500/30 bg-rose-500/10 text-rose-300', bar: 'bg-rose-400' }
}

/* ---------- Inline icons (no icon library) ---------- */
type IconProps = { className?: string }
const base = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  viewBox: '0 0 24 24',
  'aria-hidden': true,
}

const SparkIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z" />
    <path d="M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z" />
  </svg>
)
const ShieldIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6l8-3z" />
    <path d="M9 12l2 2 4-4" />
  </svg>
)
const DocIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5z" />
    <path d="M14 3v5h5M9 13h6M9 17h6" />
  </svg>
)
const WarnIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" />
    <path d="M12 9v4M12 17h.01" />
  </svg>
)
const AlertIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v5M12 16h.01" />
  </svg>
)
const SendIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
)
const ArrowRightIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
)
const BookIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 5a2 2 0 012-2h13v15H6a2 2 0 00-2 2V5z" />
    <path d="M4 20a2 2 0 002 1h13" />
  </svg>
)
const CubeIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 2l9 5v10l-9 5-9-5V7l9-5z" />
    <path d="M12 22V12M21 7l-9 5-9-5" />
  </svg>
)
const CompareIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 7h14M14 3l4 4-4 4" />
    <path d="M20 17H6M10 13l-4 4 4 4" />
  </svg>
)
const TrendIcon = ({ className }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M3 17l6-6 4 4 8-8" />
    <path d="M15 7h6v6" />
  </svg>
)

const SUGGESTIONS = [
  { title: 'Fundamentals', q: 'What is Agentic AI?', Icon: BookIcon },
  { title: 'Architecture', q: 'What are the key components of Agentic AI?', Icon: CubeIcon },
  { title: 'Comparison', q: 'How do LLMs differ from AI agents?', Icon: CompareIcon },
  { title: 'Business value', q: 'What are the benefits of Agentic AI?', Icon: TrendIcon },
]

function Avatar() {
  return (
    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-accent-400/20 bg-surface-2 text-accent-400">
      <SparkIcon className="h-4 w-4" />
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  // Auto-grow the textarea (and shrink it back after sending)
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
  }, [input])

  async function send(text?: string) {
    const query = (text ?? input).trim()
    if (!query || loading) return
    setMessages((m) => [...m, { role: 'user', text: query }])
    setInput('')
    setLoading(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (!res.ok) {
        setMessages((m) => [...m, { role: 'error', text: errorText(res.status) }])
      } else {
        const data = await res.json()
        setMessages((m) => [
          ...m,
          {
            role: 'assistant',
            text: data.final_answer,
            confidence: data.confidence_score,
            grounded: data.grounded,
            sources: data.sources ?? [],
          },
        ])
      }
    } catch {
      setMessages((m) => [
        ...m,
        { role: 'error', text: 'Cannot reach the server. Is the backend running?' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const empty = messages.length === 0
  const canSend = !loading && input.trim().length > 0

  return (
    <div className="flex h-dvh w-full flex-col">
      {/* Header */}
      <header className="relative z-10 shrink-0 bg-ink/70 backdrop-blur-xl">
        <div className="mx-auto flex h-14 w-full max-w-3xl items-center justify-between gap-3 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="logo-tile flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-white">
              <SparkIcon className="h-4 w-4" />
            </div>
            <div className="min-w-0 leading-tight">
              <h1 className="truncate text-sm font-semibold tracking-tight text-ivory">
                Agentic AI eBook Assistant
              </h1>
              <p className="truncate text-[11px] text-mute">Cited answers, straight from the source</p>
            </div>
          </div>
          <span className="hidden shrink-0 items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-zinc-300 sm:inline-flex">
            <span className="h-1.5 w-1.5 rounded-full bg-glow shadow-[0_0_8px_rgba(66,200,213,0.8)]" />
            Grounded in the eBook
          </span>
        </div>
        <div className="h-px bg-gradient-to-r from-transparent via-accent-400/30 to-transparent" />
      </header>

      {/* Conversation */}
      <main className="scroll-area flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
          {empty ? (
            <section className="rise mx-auto flex max-w-2xl flex-col items-center pt-6 text-center sm:pt-14">
              <div className="relative flex h-20 w-20 items-center justify-center">
                <div className="absolute -inset-8 rounded-full bg-accent-400/10 blur-2xl" />
                <div className="absolute -inset-3 rounded-full border border-accent-400/15" />
                <div className="absolute -inset-6 rounded-full border border-accent-400/[0.07]" />
                <div className="logo-tile relative flex h-16 w-16 items-center justify-center rounded-2xl text-white">
                  <SparkIcon className="h-8 w-8" />
                </div>
              </div>
              <h2 className="font-display mt-10 text-4xl font-medium leading-tight tracking-tight text-ivory sm:text-5xl">
                Ask the eBook <span className="italic text-accent-400">anything</span>
              </h2>
              <p className="mt-4 max-w-md text-[15px] leading-relaxed text-mute">
                Get answers about Agentic AI that come only from the eBook, with a confidence score
                and the pages they were drawn from.
              </p>
              <div className="mt-9 grid w-full grid-cols-1 gap-3 text-left sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.q}
                    onClick={() => send(s.q)}
                    className="suggest group rounded-2xl p-4 text-left"
                  >
                    <span className="relative flex items-center justify-between">
                      <span className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-accent-400">
                        <s.Icon className="h-4 w-4" />
                        {s.title}
                      </span>
                      <ArrowRightIcon className="h-4 w-4 -translate-x-1 text-accent-400 opacity-0 transition duration-200 group-hover:translate-x-0 group-hover:opacity-100" />
                    </span>
                    <span className="relative mt-3 block text-[15px] leading-snug text-ivory">
                      {s.q}
                    </span>
                  </button>
                ))}
              </div>
              <p className="mt-6 text-xs text-mute">
                Page citations · Confidence scoring · Declines questions the eBook can&apos;t support
              </p>
            </section>
          ) : (
            <div role="log" aria-live="polite" aria-label="Conversation" className="space-y-6">
              {messages.map((m, i) => {
                if (m.role === 'user') {
                  return (
                    <div key={i} className="rise flex justify-end">
                      <div className="user-bubble max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md px-4 py-2.5 text-[15px] leading-relaxed text-ivory sm:max-w-[75%]">
                        {m.text}
                      </div>
                    </div>
                  )
                }

                if (m.role === 'error') {
                  return (
                    <div key={i} className="rise flex items-start gap-3">
                      <Avatar />
                      <div
                        role="alert"
                        className="flex max-w-[88%] items-start gap-2.5 rounded-2xl rounded-tl-md border border-rose-500/25 bg-rose-500/[0.08] px-4 py-3 text-sm text-rose-200 sm:max-w-[75%]"
                      >
                        <AlertIcon className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                        <span>{m.text}</span>
                      </div>
                    </div>
                  )
                }

                const refused = m.text.trim() === REFUSAL

                if (refused) {
                  return (
                    <div key={i} className="rise flex items-start gap-3">
                      <Avatar />
                      <div className="flex max-w-[88%] items-start gap-2.5 rounded-2xl rounded-tl-md border border-amber-500/25 bg-amber-500/[0.07] px-4 py-3 text-sm text-amber-100 sm:max-w-[75%]">
                        <WarnIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                        <div>
                          <p className="font-medium">{m.text}</p>
                          <p className="mt-1 text-xs text-amber-200/70">
                            Try a question about the topics the eBook covers.
                          </p>
                        </div>
                      </div>
                    </div>
                  )
                }

                const t = tone(m.confidence)
                const pct = Math.round(m.confidence * 100)
                const pages = Array.from(
                  new Map(m.sources.map((s) => [`${s.source}-${s.page}`, s])).values(),
                )

                return (
                  <div key={i} className="rise flex items-start gap-3">
                    <Avatar />
                    <div className="surface min-w-0 flex-1 overflow-hidden rounded-2xl rounded-tl-md">
                      <div className="px-4 py-4 sm:px-5">
                        <div className="md">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              table: ({ children }) => (
                                <div className="table-wrap">
                                  <table>{children}</table>
                                </div>
                              ),
                            }}
                          >
                            {m.text}
                          </ReactMarkdown>
                        </div>
                      </div>

                      <div className="space-y-3 border-t border-white/[0.06] bg-white/[0.02] px-4 py-3 sm:px-5">
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium ${t.pill}`}
                          >
                            <span className="text-[10px] uppercase tracking-wider opacity-70">
                              Confidence
                            </span>
                            <span className="tabular-nums">{pct}%</span>
                            <span className="h-1 w-12 overflow-hidden rounded-full bg-white/10">
                              <span
                                className={`block h-full rounded-full ${t.bar}`}
                                style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                              />
                            </span>
                          </span>
                          {m.grounded && (
                            <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs font-medium text-ivory">
                              <ShieldIcon className="h-3.5 w-3.5 text-glow" />
                              Grounded
                            </span>
                          )}
                        </div>

                        {pages.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="mr-1 text-[11px] font-medium uppercase tracking-wider text-mute">
                              Sources
                            </span>
                            {pages.map((s) => (
                              <span
                                key={`${s.source}-${s.page}`}
                                title={`${s.source ?? 'eBook'}${s.page != null ? `, page ${s.page}` : ''}`}
                                className="inline-flex cursor-default items-center gap-1.5 rounded-lg border border-white/10 bg-surface-2 px-2 py-1 text-xs text-zinc-300 transition duration-150 hover:-translate-y-0.5 hover:border-accent-400/50 hover:text-ivory"
                              >
                                <DocIcon className="h-3.5 w-3.5 text-accent-400/80" />
                                {s.page != null ? `p. ${s.page}` : 'eBook'}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}

              {loading && (
                <div className="rise flex items-start gap-3" role="status">
                  <Avatar />
                  <div className="surface flex items-center gap-3 rounded-2xl rounded-tl-md px-4 py-3 text-sm text-mute">
                    <span className="flex gap-1" aria-hidden="true">
                      <span className="dot h-1.5 w-1.5 rounded-full bg-accent-400" />
                      <span className="dot h-1.5 w-1.5 rounded-full bg-accent-400" />
                      <span className="dot h-1.5 w-1.5 rounded-full bg-accent-400" />
                    </span>
                    Searching the eBook…
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Composer */}
      <footer className="shrink-0 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          <div className="composer flex items-end gap-2 rounded-2xl p-2">
            <label htmlFor="question" className="sr-only">
              Ask a question about the eBook
            </label>
            <textarea
              id="question"
              ref={taRef}
              value={input}
              rows={1}
              maxLength={MAX_LEN}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="Ask about the eBook…"
              className="max-h-40 min-w-0 flex-1 resize-none bg-transparent px-3 py-2 text-[15px] leading-relaxed text-ivory outline-none placeholder:text-mute"
            />
            <button
              onClick={() => send()}
              disabled={!canSend}
              aria-label="Send question"
              className="send-btn inline-flex h-10 shrink-0 items-center gap-2 rounded-xl px-3 text-sm font-semibold sm:px-4"
            >
              <span className="hidden sm:inline">Send</span>
              <SendIcon className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-2 flex items-center justify-between px-1 text-[11px] text-mute">
            <span>Enter to send · Shift+Enter for a new line</span>
            {input.length > 800 && (
              <span className={input.length >= MAX_LEN ? 'text-rose-400' : ''}>
                {input.length}/{MAX_LEN}
              </span>
            )}
          </div>
        </div>
      </footer>
    </div>
  )
}
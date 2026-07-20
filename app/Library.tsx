"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { formatTime } from "./pitch-utils";
import {
  BUILTIN_CARDS,
  fetchJob,
  fetchLibrary,
  fileToBackground,
  getBackground,
  pipelineAvailable,
  setBackground,
  STEP_LABELS,
  uploadSong,
  type JobStatus,
  type LibraryCard,
} from "./library-data";

const UPLOAD_STEPS = ["separating", "isolating", "tracking", "transcribing", "finalizing"];

function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="8.5" cy="9.5" r="1.7" fill="currentColor" />
      <path d="M4 17l4.5-4.5 3.5 3 3-2.5L20 17" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
    </svg>
  );
}

function SongCard({
  card,
  background,
  onOpen,
  onPickBackground,
  onClearBackground,
}: {
  card: LibraryCard;
  background: string | null;
  onOpen: (card: LibraryCard) => void;
  onPickBackground: (card: LibraryCard, file: File) => void;
  onClearBackground: (card: LibraryCard) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const initials = card.title.replace(/[^A-Za-z0-9]/g, "").slice(0, 2).toUpperCase() || "♪";

  return (
    <article className="songCard">
      <button
        type="button"
        className={`songCardArt ${background ? "hasImage" : ""}`}
        style={background ? { backgroundImage: `url(${background})` } : undefined}
        onClick={() => onOpen(card)}
        aria-label={`Practice ${card.title} by ${card.artist || "unknown artist"}`}
      >
        {!background && <span className="songCardDisc" aria-hidden="true">{initials}</span>}
        <span className="playOverlay" aria-hidden="true">
          <span className="playIcon" />
        </span>
        <span className="durationBadge">{formatTime(card.duration)}</span>
        {!card.builtin && <span className="uploadTag">Uploaded</span>}
      </button>
      <div className="songCardMeta">
        <div className="songCardText">
          <strong title={card.title}>{card.title}</strong>
          <span title={card.artist}>{card.artist || "Unknown artist"}</span>
        </div>
        <div className="songCardActions">
          <button
            type="button"
            className="cardIconButton"
            onClick={() => inputRef.current?.click()}
            aria-label={background ? `Change background for ${card.title}` : `Add background for ${card.title}`}
          >
            <ImageIcon />
          </button>
          {background && (
            <button
              type="button"
              className="cardClearButton"
              onClick={() => onClearBackground(card)}
              aria-label={`Remove background for ${card.title}`}
            >
              Clear
            </button>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            const chosen = event.target.files?.[0];
            if (chosen) onPickBackground(card, chosen);
            event.target.value = "";
          }}
        />
      </div>
    </article>
  );
}

export default function Library({ onOpen, onTraining }: { onOpen: (card: LibraryCard) => void; onTraining: () => void }) {
  // Seed with the built-in songs so they render on the server with no fetch;
  // uploaded songs from the manifest are merged in after mount.
  const [cards, setCards] = useState<LibraryCard[]>(BUILTIN_CARDS);
  const [backgrounds, setBackgrounds] = useState<Record<string, string | null>>({});
  const [serverUp, setServerUp] = useState<boolean | null>(null);

  const [uploadOpen, setUploadOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [artist, setArtist] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [job, setJob] = useState<JobStatus | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    const list = await fetchLibrary();
    const nextBackgrounds: Record<string, string | null> = {};
    for (const card of list) nextBackgrounds[card.id] = getBackground(card.id);
    setCards(list);
    setBackgrounds(nextBackgrounds);
    return list;
  }, []);

  const checkPipeline = useCallback(async () => {
    const available = await pipelineAvailable();
    setServerUp(available);
    return available;
  }, []);

  useEffect(() => {
    // Initial library load + companion health probe. Both setState calls happen
    // only after their awaited fetch resolves, not synchronously in this effect.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async client-only load
    void refresh();
    void checkPipeline();
    return () => {
      if (pollRef.current) window.clearTimeout(pollRef.current);
    };
  }, [checkPipeline, refresh]);

  const pickBackground = useCallback(async (card: LibraryCard, chosen: File) => {
    try {
      const dataUrl = await fileToBackground(chosen);
      setBackground(card.id, dataUrl);
      setBackgrounds((current) => ({ ...current, [card.id]: dataUrl }));
    } catch {
      // Ignore unreadable images; the card keeps its current art.
    }
  }, []);

  const clearBackground = useCallback((card: LibraryCard) => {
    setBackground(card.id, null);
    setBackgrounds((current) => ({ ...current, [card.id]: null }));
  }, []);

  const resetUpload = useCallback(() => {
    if (pollRef.current) window.clearTimeout(pollRef.current);
    setFile(null);
    setTitle("");
    setArtist("");
    setUploadError("");
    setJob(null);
  }, []);

  const closeUpload = useCallback(() => {
    setUploadOpen(false);
    resetUpload();
  }, [resetUpload]);

  const openUpload = useCallback(() => {
    setUploadOpen(true);
    setServerUp(null);
    void checkPipeline();
  }, [checkPipeline]);

  const poll = useCallback(
    (jobId: string) => {
      const tick = async () => {
        try {
          const status = await fetchJob(jobId);
          setJob(status);
          if (status.status === "done") {
            await refresh();
            return;
          }
          if (status.status === "error") return;
          pollRef.current = window.setTimeout(tick, 1200);
        } catch {
          setJob((current) =>
            current
              ? { ...current, status: "error", error: "Lost connection to the pipeline server." }
              : current,
          );
        }
      };
      void tick();
    },
    [refresh],
  );

  const submitUpload = useCallback(async () => {
    if (!file) {
      setUploadError("Choose an mp3 file first.");
      return;
    }
    if (file.size < 1024) {
      setUploadError(
        `"${file.name}" is empty or truncated (${file.size} bytes) — the download that produced it likely failed. Re-download the mp3 and pick it again.`,
      );
      return;
    }
    if (!title.trim()) {
      setUploadError("Give the song a title.");
      return;
    }
    setUploadError("");
    try {
      if (!(await checkPipeline())) {
        setUploadError(
          "The pipeline is still unreachable from this page. Open localhost:3000 and try again.",
        );
        return;
      }
      const { jobId, songId } = await uploadSong(file, title.trim(), artist.trim());
      setJob({
        jobId,
        songId,
        status: "queued",
        step: null,
        steps: [],
        stepOrder: UPLOAD_STEPS,
        duration: 0,
        error: null,
      });
      poll(jobId);
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? `${error.message} — is the pipeline server running? (npm run pipeline)`
          : "Upload failed.",
      );
    }
  }, [artist, checkPipeline, file, poll, title]);

  const playProcessed = useCallback(() => {
    if (!job) return;
    const match = cards.find((card) => card.id === job.songId);
    if (match) {
      closeUpload();
      onOpen(match);
    }
  }, [cards, closeUpload, job, onOpen]);

  const currentStepIndex = job?.step ? UPLOAD_STEPS.indexOf(job.step) : -1;
  const processing = job?.status === "queued" || job?.status === "running";

  return (
    <main className="appShell">
      <header className="topbar">
        <span className="brand">so<span>na</span></span>
        <nav className="mainNav" aria-label="Main navigation">
          <button type="button" className="active">Songs</button>
          <button type="button" onClick={onTraining}>Training</button>
        </nav>
        <span className="topbarSpacer" aria-hidden="true" />
      </header>

      <section className="libraryHead">
        <div>
          <span className="eyebrow">Your library</span>
          <h1 className="libraryTitle">Choose a song to practice</h1>
        </div>
        <p className="libraryCount">
          {cards.length} {cards.length === 1 ? "song" : "songs"}
        </p>
      </section>

      <div className="libraryGrid">
        {cards.map((card) => (
          <SongCard
            key={card.id}
            card={card}
            background={backgrounds[card.id] ?? null}
            onOpen={onOpen}
            onPickBackground={pickBackground}
            onClearBackground={clearBackground}
          />
        ))}

        <button type="button" className="addCard" onClick={openUpload}>
          <span className="addPlus" aria-hidden="true">+</span>
          <span className="addLabel">Add a song</span>
          <span className="addHint">Upload an mp3 and we&apos;ll transcribe it</span>
        </button>
      </div>

      {uploadOpen && (
        <div className="uploadBackdrop" role="dialog" aria-modal="true" aria-label="Add a song">
          <div className="uploadPanel">
            <div className="uploadPanelHead">
              <h2>Add a song</h2>
              <button type="button" className="uploadClose" onClick={closeUpload} aria-label="Close">
                ×
              </button>
            </div>

            {!job && (
              <>
                {serverUp === false && (
                  <p className="uploadNotice">
                    The pipeline server isn&apos;t responding. Use the local app at{" "}
                    the address printed by <code>npm run dev</code>, keep{" "}
                    <code>npm run pipeline</code> running, and allow local-network
                    access if your browser asks.
                  </p>
                )}
                <label className="uploadField">
                  <span>MP3 file</span>
                  <input
                    type="file"
                    accept="audio/mpeg,.mp3"
                    onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                  />
                </label>
                {file && <p className="uploadFileName">{file.name}</p>}
                <label className="uploadField">
                  <span>Title</span>
                  <input
                    type="text"
                    value={title}
                    placeholder="Song title"
                    onChange={(event) => setTitle(event.target.value)}
                  />
                </label>
                <label className="uploadField">
                  <span>Artist <em>(optional)</em></span>
                  <input
                    type="text"
                    value={artist}
                    placeholder="Artist"
                    onChange={(event) => setArtist(event.target.value)}
                  />
                </label>
                {uploadError && <p className="uploadError">{uploadError}</p>}
                <div className="uploadButtons">
                  <button type="button" className="ghostButton" onClick={closeUpload}>
                    Cancel
                  </button>
                  <button type="button" className="primaryButton" onClick={submitUpload}>
                    Process song
                  </button>
                </div>
                <p className="uploadFootnote">
                  Runs Kim stem separation, lead-vocal isolation, pitch tracking, and transcription locally. Takes a few minutes.
                </p>
              </>
            )}

            {job && processing && (
              <div className="jobProgress">
                <p className="jobHeadline">Transcribing “{title || job.songId}”…</p>
                <ol className="jobSteps">
                  {UPLOAD_STEPS.map((step, index) => {
                    const state =
                      currentStepIndex < 0
                        ? index === 0
                          ? "active"
                          : "pending"
                        : index < currentStepIndex
                          ? "done"
                          : index === currentStepIndex
                            ? "active"
                            : "pending";
                    return (
                      <li key={step} className={`jobStep ${state}`}>
                        <span className="jobDot" aria-hidden="true" />
                        {STEP_LABELS[step] ?? step}
                      </li>
                    );
                  })}
                </ol>
                <p className="uploadFootnote">Keep this tab open — you can watch or wait.</p>
              </div>
            )}

            {job && job.status === "done" && (
              <div className="jobProgress">
                <p className="jobHeadline jobDone">“{job.title || title}” is ready 🎉</p>
                <div className="uploadButtons">
                  <button type="button" className="ghostButton" onClick={closeUpload}>
                    Back to library
                  </button>
                  <button type="button" className="primaryButton" onClick={playProcessed}>
                    Practice now
                  </button>
                </div>
              </div>
            )}

            {job && job.status === "error" && (
              <div className="jobProgress">
                <p className="jobHeadline jobFailed">Processing failed</p>
                <p className="uploadError">{job.error || "The pipeline did not finish."}</p>
                <div className="uploadButtons">
                  <button type="button" className="ghostButton" onClick={closeUpload}>
                    Close
                  </button>
                  <button type="button" className="primaryButton" onClick={resetUpload}>
                    Try again
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

"use client";

import { useCallback, useState } from "react";
import Library from "./Library";
import StudioPlayer from "./StudioPlayer";
import VocalTraining from "./VocalTraining";
import { resolveSong, type LibraryCard } from "./library-data";
import type { Song } from "./songs";

export default function Home() {
  const [section, setSection] = useState<"songs" | "training">("songs");
  const [song, setSong] = useState<Song | null>(null);
  const [loadingCard, setLoadingCard] = useState<LibraryCard | null>(null);
  const [error, setError] = useState("");

  const open = useCallback(async (card: LibraryCard) => {
    setError("");
    setLoadingCard(card);
    try {
      setSong(await resolveSong(card));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load that song.");
    } finally {
      setLoadingCard(null);
    }
  }, []);

  const back = useCallback(() => setSong(null), []);

  const navigate = useCallback((next: "songs" | "training") => {
    setSong(null);
    setSection(next);
  }, []);

  if (song) return <StudioPlayer key={song.id} song={song} onBack={back} />;
  if (section === "training") return <VocalTraining onNavigate={navigate} />;

  return (
    <>
      <Library onOpen={open} onTraining={() => navigate("training")} />
      {loadingCard && (
        <div className="modalBackdrop">
          <div className="loadingNote">Loading “{loadingCard.title}”…</div>
        </div>
      )}
      {error && (
        <div className="modalBackdrop" onClick={() => setError("")}>
          <div className="loadingNote loadingError">{error}</div>
        </div>
      )}
    </>
  );
}

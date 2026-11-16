import { renderTake, type TakeRenderRequest } from "./studio-utils";

// Rendering and WAV-encoding a full take touches several hundred megabytes of
// samples. Doing it here keeps the studio responsive while files are prepared.
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<TakeRenderRequest>) => void) | null;
  postMessage(message: unknown): void;
};

// Announce readiness before any samples are handed over, so the page can
// fall back to rendering inline if this script never loads.
scope.postMessage({ ready: true });

scope.onmessage = (event) => {
  try {
    scope.postMessage({ ok: true, result: renderTake(event.data) });
  } catch (error) {
    scope.postMessage({
      ok: false,
      error: error instanceof Error ? error.message : "The take could not be rendered.",
    });
  }
};

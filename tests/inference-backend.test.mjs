import assert from "node:assert/strict";
import test from "node:test";

import { selectInferenceBackend } from "../scripts/select-inference-backend.mjs";

test("selects the fastest packaged backend for each desktop host", () => {
  assert.equal(selectInferenceBackend({ platform: "darwin" }).backend, "coreml");
  assert.equal(
    selectInferenceBackend({ platform: "linux", nvidiaAvailable: true }).backend,
    "cuda",
  );
  assert.equal(
    selectInferenceBackend({ platform: "win32", nvidiaAvailable: true }).backend,
    "cuda",
  );
  assert.equal(selectInferenceBackend({ platform: "win32" }).backend, "directml");
  assert.equal(selectInferenceBackend({ platform: "linux" }).backend, "cpu");
});

test("honors supported explicit overrides and rejects impossible hosts", () => {
  assert.equal(
    selectInferenceBackend({ platform: "linux", requested: "cpu", nvidiaAvailable: true })
      .backend,
    "cpu",
  );
  assert.throws(
    () => selectInferenceBackend({ platform: "linux", requested: "directml" }),
    /only on Windows/,
  );
  assert.throws(
    () => selectInferenceBackend({ platform: "win32", requested: "coreml" }),
    /only on macOS/,
  );
});

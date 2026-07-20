/** Host-aware ONNX Runtime package selection used by pipeline setup. */

const BACKENDS = {
  cpu: { packageName: "onnxruntime", version: "1.27.0" },
  coreml: { packageName: "onnxruntime", version: "1.27.0" },
  cuda: { packageName: "onnxruntime-gpu", version: "1.27.0" },
  // DirectML currently ships on its own release cadence.
  directml: { packageName: "onnxruntime-directml", version: "1.24.4" },
};

export function selectInferenceBackend({
  platform = process.platform,
  requested = process.env.SONA_ACCELERATOR || "auto",
  nvidiaAvailable = false,
} = {}) {
  const normalized = requested.trim().toLowerCase() || "auto";
  let backend = normalized;
  if (normalized === "auto") {
    if (nvidiaAvailable && platform !== "darwin") backend = "cuda";
    else if (platform === "darwin") backend = "coreml";
    else if (platform === "win32") backend = "directml";
    else backend = "cpu";
  }
  if (backend === "dml") backend = "directml";
  if (!(backend in BACKENDS)) {
    throw new Error(
      `Unsupported SONA_ACCELERATOR=${JSON.stringify(requested)} for automatic setup; ` +
      "choose auto, cpu, coreml, cuda, or directml.",
    );
  }
  if (backend === "coreml" && platform !== "darwin") {
    throw new Error("CoreML setup is available only on macOS.");
  }
  if (backend === "directml" && platform !== "win32") {
    throw new Error("DirectML setup is available only on Windows.");
  }
  return { backend, ...BACKENDS[backend] };
}

export const ACCELERATED_ONNX_PROVIDERS = new Set([
  "TensorrtExecutionProvider",
  "CUDAExecutionProvider",
  "MIGraphXExecutionProvider",
  "ROCMExecutionProvider",
  "DmlExecutionProvider",
  "CoreMLExecutionProvider",
  "OpenVINOExecutionProvider",
  "QNNExecutionProvider",
  "WebGPUExecutionProvider",
]);

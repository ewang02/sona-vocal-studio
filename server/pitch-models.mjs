/**
 * Pitch models a song can be analysed with. The tuning of each lives with
 * the pipeline in work/pitch_models.py; this list only names what the
 * companion will accept (a pipeline test keeps the two in step).
 */
export const PITCH_MODELS = ["crepe", "rmvpe", "swiftf0"];
export const DEFAULT_PITCH_MODEL = "crepe";

/** Normalise a requested model; null when it is not one Sona offers. */
export function resolvePitchModel(value) {
  const name = String(value ?? "").trim().toLowerCase() || DEFAULT_PITCH_MODEL;
  return PITCH_MODELS.includes(name) ? name : null;
}

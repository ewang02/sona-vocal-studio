const DEPLOYED_ORIGINS = new Set([
  "https://sona-vocal-studio.edwardwang070206.chatgpt.site",
]);

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

function configuredOrigins(value = process.env.PIPELINE_ALLOWED_ORIGINS || "") {
  return new Set(
    value
      .split(",")
      .map((origin) => origin.trim())
      .filter(Boolean)
      .map((origin) => {
        try {
          return new URL(origin).origin;
        } catch {
          return null;
        }
      })
      .filter(Boolean),
  );
}

/** Return the normalized origin only when it is allowed to control the pipeline. */
export function allowedPipelineOrigin(
  value,
  additionalOrigins = process.env.PIPELINE_ALLOWED_ORIGINS,
) {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (
      DEPLOYED_ORIGINS.has(url.origin) ||
      configuredOrigins(additionalOrigins).has(url.origin)
    ) {
      return url.origin;
    }
    if (
      (url.protocol === "http:" || url.protocol === "https:") &&
      LOOPBACK_HOSTS.has(url.hostname) &&
      !url.username &&
      !url.password
    ) {
      return url.origin;
    }
  } catch {
    // Invalid Origin headers are denied below.
  }
  return null;
}

/** Apply standard CORS plus the browser opt-in for public-to-local requests. */
export function applyPipelineCors(request, response) {
  const origin = allowedPipelineOrigin(request.headers.origin);
  response.setHeader(
    "Vary",
    "Origin, Access-Control-Request-Private-Network",
  );
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (!origin) return;

  response.setHeader("Access-Control-Allow-Origin", origin);
  if (request.headers["access-control-request-private-network"] === "true") {
    response.setHeader("Access-Control-Allow-Private-Network", "true");
  }
}

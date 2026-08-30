// No hosted origin is trusted by default. A deployment opts in explicitly with
// PIPELINE_ALLOWED_ORIGINS so a companion never answers a site the user did not
// name. Loopback development origins are allowed separately, below.
const DEPLOYED_ORIGINS = new Set();

const LOCAL_DEVELOPMENT_ORIGINS = new Set([
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

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
    if (LOCAL_DEVELOPMENT_ORIGINS.has(url.origin)) return url.origin;
  } catch {
    // Invalid Origin headers are denied below.
  }
  return null;
}

/** Apply standard CORS plus the browser opt-in for public-to-local requests. */
export function applyPipelineCors(request, response) {
  const origin = allowedPipelineOrigin(request.headers.origin);
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader(
    "Vary",
    "Origin, Access-Control-Request-Private-Network",
  );
  response.setHeader("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type, Range, X-Sona-Action");
  response.setHeader(
    "Access-Control-Expose-Headers",
    "Accept-Ranges, Content-Length, Content-Range",
  );
  if (!origin) return;

  response.setHeader("Access-Control-Allow-Origin", origin);
  if (request.headers["access-control-request-private-network"] === "true") {
    response.setHeader("Access-Control-Allow-Private-Network", "true");
  }
}

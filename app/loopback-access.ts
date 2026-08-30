/**
 * Rules for reaching the loopback-only companion from the page.
 *
 * Kept free of other imports so it stays directly testable under Node.
 */

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

/** Local pages can connect silently; hosted pages wait for an explicit user action. */
export function shouldAutoConnectForHostname(hostname: string): boolean {
  return LOOPBACK_HOSTS.has(hostname);
}

/**
 * Safari still treats http://127.0.0.1 as blocked mixed content on an HTTPS
 * page, unlike Chrome and Firefox, which follow the Secure Contexts rule that
 * loopback is potentially trustworthy (WebKit bug 171934). The companion is
 * then unreachable no matter how it is configured, so say that instead of
 * blaming a companion that is running fine. Returns null when the hosted page
 * is expected to connect after the browser's local-network prompt.
 */
export function describeHostedLoopbackBlock(
  protocol: string,
  hostname: string,
  userAgent: string,
): string | null {
  if (shouldAutoConnectForHostname(hostname) || protocol !== "https:") return null;
  const webkit = /\bAppleWebKit\b/.test(userAgent);
  const chromiumOrGecko = /\b(?:Chrome|Chromium|CriOS|FxiOS|Edg|Gecko\/)\b/.test(userAgent);
  if (!webkit || chromiumOrGecko) return null;
  return "This browser blocks a secure page from reaching your local library. Open Sona from http://localhost:3000 instead, or use Chrome, Edge, or Firefox.";
}

/** The same check against the live page. Null while server-rendering. */
export function hostedLoopbackBlock(): string | null {
  if (typeof window === "undefined") return null;
  return describeHostedLoopbackBlock(
    window.location.protocol,
    window.location.hostname,
    window.navigator.userAgent,
  );
}

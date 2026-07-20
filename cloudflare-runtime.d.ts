interface Fetcher {
  fetch(request: Request): Promise<Response>;
}

type D1Database = unknown;

declare module "cloudflare:workers" {
  export const env: Record<string, unknown>;
}

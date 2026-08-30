import assert from "node:assert/strict";
import test from "node:test";

import {
  allowedPipelineOrigin,
  applyPipelineCors,
} from "../server/pipeline-cors.mjs";

test("allows only exact local development origins by default", () => {
  // No hosted origin is trusted until the operator names one.
  assert.equal(allowedPipelineOrigin("https://sona.example.com"), null);
  assert.equal(allowedPipelineOrigin("http://localhost:3000"), "http://localhost:3000");
  assert.equal(allowedPipelineOrigin("http://127.0.0.1:3000"), "http://127.0.0.1:3000");
  assert.equal(allowedPipelineOrigin("http://localhost:3001"), null);
  assert.equal(allowedPipelineOrigin("https://attacker.example"), null);
  assert.equal(allowedPipelineOrigin("not an origin"), null);
});

test("allows explicitly configured distribution origins", () => {
  assert.equal(
    allowedPipelineOrigin(
      "https://studio.example.com",
      "https://studio.example.com, https://another.example",
    ),
    "https://studio.example.com",
  );
  assert.equal(
    allowedPipelineOrigin("https://attacker.example", "https://studio.example.com"),
    null,
  );
});

test("opts an allowed public origin into local-network preflight", () => {
  const headers = {};
  const response = { setHeader: (name, value) => { headers[name] = value; } };
  process.env.PIPELINE_ALLOWED_ORIGINS = "https://studio.example.com";
  try {
    applyPipelineCors(
      {
        headers: {
          origin: "https://studio.example.com",
          "access-control-request-private-network": "true",
        },
      },
      response,
    );
  } finally {
    delete process.env.PIPELINE_ALLOWED_ORIGINS;
  }
  assert.equal(headers["Access-Control-Allow-Origin"], "https://studio.example.com");
  assert.equal(headers["Access-Control-Allow-Private-Network"], "true");
  assert.match(headers["Access-Control-Allow-Methods"], /HEAD/);
  assert.match(headers["Access-Control-Allow-Headers"], /Range/);
  assert.match(headers["Access-Control-Allow-Headers"], /X-Sona-Action/);
  assert.match(headers["Access-Control-Expose-Headers"], /Content-Range/);
});

test("does not grant CORS or local-network access to another website", () => {
  const headers = {};
  const response = { setHeader: (name, value) => { headers[name] = value; } };
  applyPipelineCors(
    {
      headers: {
        origin: "https://attacker.example",
        "access-control-request-private-network": "true",
      },
    },
    response,
  );
  assert.equal(headers["Access-Control-Allow-Origin"], undefined);
  assert.equal(headers["Access-Control-Allow-Private-Network"], undefined);
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  allowedPipelineOrigin,
  applyPipelineCors,
} from "../server/pipeline-cors.mjs";

test("allows only the deployed app and loopback development origins", () => {
  assert.equal(
    allowedPipelineOrigin("https://sona-vocal-studio.edwardwang070206.chatgpt.site"),
    "https://sona-vocal-studio.edwardwang070206.chatgpt.site",
  );
  assert.equal(allowedPipelineOrigin("http://localhost:3000"), "http://localhost:3000");
  assert.equal(allowedPipelineOrigin("http://127.0.0.1:3001"), "http://127.0.0.1:3001");
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
  applyPipelineCors(
    {
      headers: {
        origin: "https://sona-vocal-studio.edwardwang070206.chatgpt.site",
        "access-control-request-private-network": "true",
      },
    },
    response,
  );
  assert.equal(
    headers["Access-Control-Allow-Origin"],
    "https://sona-vocal-studio.edwardwang070206.chatgpt.site",
  );
  assert.equal(headers["Access-Control-Allow-Private-Network"], "true");
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

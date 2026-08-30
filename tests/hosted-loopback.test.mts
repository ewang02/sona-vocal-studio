import assert from "node:assert/strict";
import test from "node:test";
import {
  describeHostedLoopbackBlock,
  shouldAutoConnectForHostname,
} from "../app/loopback-access.ts";

const SAFARI =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15";
const IOS_SAFARI =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Mobile/15E148 Safari/604.1";
const CHROME =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36";
const EDGE =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0";
const FIREFOX =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0";
const CHROME_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 18_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/142.0.0.0 Mobile/15E148 Safari/604.1";

test("only loopback pages connect to the companion without a user gesture", () => {
  assert.equal(shouldAutoConnectForHostname("127.0.0.1"), true);
  assert.equal(shouldAutoConnectForHostname("localhost"), true);
  assert.equal(shouldAutoConnectForHostname("[::1]"), true);
  assert.equal(shouldAutoConnectForHostname("studio.example.com"), false);
});

test("hosted WebKit pages are told the browser blocks loopback, not the companion", () => {
  // Safari blocks https -> http://127.0.0.1 as mixed content (WebKit bug 171934).
  for (const agent of [SAFARI, IOS_SAFARI]) {
    const message = describeHostedLoopbackBlock("https:", "studio.example.com", agent);
    assert.match(message ?? "", /blocks a secure page/);
    assert.match(message ?? "", /localhost:3000/);
  }

  // Browsers that follow the Secure Contexts loopback rule reach the companion
  // after their own local-network prompt, so they must not see the warning.
  for (const agent of [CHROME, EDGE, FIREFOX, CHROME_IOS]) {
    assert.equal(describeHostedLoopbackBlock("https:", "studio.example.com", agent), null);
  }

  // A locally served page is same-scheme loopback and never blocked.
  assert.equal(describeHostedLoopbackBlock("http:", "127.0.0.1", SAFARI), null);
  assert.equal(describeHostedLoopbackBlock("http:", "localhost", SAFARI), null);
  // A plain-http hosted page is not a mixed-content case either.
  assert.equal(describeHostedLoopbackBlock("http:", "studio.example.com", SAFARI), null);
});

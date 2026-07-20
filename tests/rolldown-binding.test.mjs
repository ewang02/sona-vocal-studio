import assert from "node:assert/strict";
import test from "node:test";

import {
  bindingNameFor,
  detectLinuxLibc,
} from "../scripts/ensure-rolldown-binding.mjs";

test("maps supported desktop and Cloudflare platforms to Rolldown bindings", () => {
  assert.equal(bindingNameFor("darwin", "arm64"), "@rolldown/binding-darwin-arm64");
  assert.equal(bindingNameFor("darwin", "x64"), "@rolldown/binding-darwin-x64");
  assert.equal(bindingNameFor("win32", "x64"), "@rolldown/binding-win32-x64-msvc");
  assert.equal(bindingNameFor("win32", "arm64"), "@rolldown/binding-win32-arm64-msvc");
  assert.equal(bindingNameFor("linux", "x64", "gnu"), "@rolldown/binding-linux-x64-gnu");
  assert.equal(bindingNameFor("linux", "x64", "musl"), "@rolldown/binding-linux-x64-musl");
  assert.equal(bindingNameFor("linux", "arm64", "gnu"), "@rolldown/binding-linux-arm64-gnu");
  assert.equal(bindingNameFor("linux", "arm64", "musl"), "@rolldown/binding-linux-arm64-musl");
});

test("detects GNU and musl Linux reports", () => {
  assert.equal(detectLinuxLibc({ header: { glibcVersionRuntime: "2.36" } }), "gnu");
  assert.equal(
    detectLinuxLibc({ header: {}, sharedObjects: ["/lib/ld-musl-x86_64.so.1"] }),
    "musl",
  );
});

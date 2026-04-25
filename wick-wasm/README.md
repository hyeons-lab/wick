# wick-wasm

`wasm-bindgen` browser / Node bindings for the
[wick](https://github.com/hyeons-lab/wick) inference engine.

> Status: pre-1.0. Today's surface covers manifest parsing, model
> loading (CPU-only), metadata access, and tokenizer encode/decode.
> `Session` / streaming generation lands in a follow-up release.

## Install

```sh
npm install @hyeonslab/wick-wasm  # not yet published
```

For now, download the latest `pkg/` artifact from the
[CI run](https://github.com/hyeons-lab/wick/actions/workflows/ci.yml)
on the most recent `main` build and install it locally:

```sh
npm install /path/to/downloaded/pkg
```

## Usage

This package is built with `wasm-pack --target bundler`, so the
`.wasm` is loaded automatically by your bundler (webpack 5+, Vite,
Rollup with `@rollup/plugin-wasm`). No manual `init()` call.

### Manifest parsing

```js
import { wickVersion, Manifest } from '@hyeonslab/wick-wasm';

console.log(wickVersion());  // e.g. "0.1.0"

const res = await fetch('/path/to/manifest.json');
const bytes = new Uint8Array(await res.arrayBuffer());
const manifest = Manifest.parse(bytes);

console.log(manifest.inferenceType);   // "llama.cpp/text-to-text"
console.log(manifest.modelUrl);        // "https://.../model.gguf"
console.log(manifest.schemaVersion);   // "1.0.0"
```

### Loading a model + tokenizing

```js
import { WickEngine } from '@hyeonslab/wick-wasm';

// Fetch the GGUF (use the `modelUrl` from a parsed manifest, or a
// direct URL).
const res = await fetch('/path/to/model.gguf');
const bytes = new Uint8Array(await res.arrayBuffer());

// Construct the engine. Optional `contextSize` defaults to 4096.
// Backend is forced to CPU on wasm.
const engine = WickEngine.fromGgufBytes(bytes, 2048);

console.log(engine.architecture);     // "lfm2"
console.log(engine.maxSeqLen);        // 4096
console.log(engine.quantization);     // "Q4_0"
console.log(engine.hasChatTemplate);  // true / false

// Tokenize a string.
const tok = engine.tokenizer;
const ids = tok.encode('hello world');  // Uint32Array
console.log(ids);
console.log(tok.decode(ids));           // round-trips to "hello world"

// Optional: render a chat template (use a JS Jinja runtime like
// `nunjucks` to apply `tok.chatTemplate` against your message list).
console.log(tok.chatTemplate);

// Release the model bytes from wasm memory when done.
engine.free();
```

> **Memory note:** `WickEngine` keeps the entire GGUF resident in
> wasm linear memory. Always `engine.free()` (or use the
> `[Symbol.dispose]()` pattern with `using` in TC39 explicit
> resource management) when you're done — otherwise the model
> stays alive until the page unloads.

`Session` (the actual `generate(...)` call with token streaming) is
**not yet exposed** — that surface needs an async-streaming design
that lands in a follow-up release. For now this package is useful
for: client-side token counting, chat-template rendering before an
API call, and inspecting model metadata before deciding to download.

Bundlers without native wasm support need a loader plugin; see the
[`wasm-pack` bundler guide](https://rustwasm.github.io/docs/wasm-pack/tutorials/npm-browser-packages/getting-started.html)
for webpack / Rollup / Parcel specifics.

If you need a no-bundler workflow (`<script type="module">` directly
in the browser, or a Node script using `import`), wait for the
follow-up `--target web` / `--target nodejs` builds — they'll ship as
sibling packages.

## Building from source

```sh
just wasm  # produces wick-wasm/pkg/
```

Requires `wasm-pack` (`cargo install wasm-pack`) and `wasm-opt`
(macOS: `brew install binaryen`; linux: `apt-get install binaryen`).

## License

Apache-2.0 OR MIT, matching the rest of the wick workspace.

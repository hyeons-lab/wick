# wick-wasm

`wasm-bindgen` browser / Node bindings for the
[wick](https://github.com/hyeons-lab/wick) inference engine.

> Status: pre-1.0. Today's surface covers manifest parsing, model
> loading (CPU-only), metadata access, tokenizer encode/decode, and
> sync streaming text generation via `Session.generate(opts, cb)`.

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
console.log(engine.quantization);     // "Q4_0", "Q8_0", "BF16", etc.
console.log(engine.hasChatTemplate);  // true / false
console.log(engine.addBosToken);      // honor when hand-building token sequences

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

### Inference (text)

```js
import { WickEngine, GenerateOpts } from '@hyeonslab/wick-wasm';

const engine = WickEngine.fromGgufBytes(gguf, 2048);
const tok = engine.tokenizer;
const session = engine.newSession();

// Seed the conversation. Use `session.appendText(prompt)` for the
// common case (tokenizer is invoked internally), or
// `session.appendTokens(ids)` when you need control over BOS/EOS.
session.appendText('Hello, what is the capital of France?');

// Configure decoding. All fields default to wick's native defaults
// (max 256 tokens, temperature 0.7, top-p 0.9, top-k 40, no stops,
// flush every 16 tokens or 50 ms).
const opts = new GenerateOpts();
opts.maxTokens = 64;
opts.temperature = 0.0;
// `tok.eosToken` is `number | undefined`. `new Uint32Array([undefined])`
// silently coerces to `0` — which would stop decoding the moment
// token 0 is produced. Always guard the lookup.
if (tok.eosToken != null) {
    opts.stopTokens = new Uint32Array([tok.eosToken]);
}

// Stream tokens as they decode. The callback fires per flush
// boundary (every `flushEveryTokens` decoded tokens, OR every
// `flushEveryMs` ms — whichever hits first) with just the *new*
// tokens, not the cumulative buffer.
let acc = [];
const summary = session.generate(opts, (newTokens) => {
    acc.push(...newTokens);
    process.stdout.write(tok.decode(newTokens));  // streaming text
});

console.log('\n---');
console.log('finish:', summary.finishReason);          // "Stop" | "MaxTokens" | ...
console.log('tokens:', summary.tokensGenerated);
console.log('decode ms:', summary.decodeMs);

session.free();
engine.free();
```

> **Worker note:** `Session.generate` is **synchronous** and blocks
> the thread it runs on for the full decode duration (potentially
> seconds). On the browser main thread that freezes the page —
> always call from a Web Worker:
>
> ```js
> // worker.js
> import { WickEngine, GenerateOpts } from '@hyeonslab/wick-wasm';
> self.onmessage = async (ev) => {
>     const engine = WickEngine.fromGgufBytes(ev.data.gguf);
>     const session = engine.newSession();
>     session.appendText(ev.data.prompt);
>     const opts = new GenerateOpts();
>     opts.maxTokens = 128;
>     session.generate(opts, (toks) => self.postMessage({ kind: 'tokens', toks }));
>     self.postMessage({ kind: 'done' });
> };
> ```
>
> On Node the sync call also blocks the JS event loop — libuv's
> background I/O thread pool keeps running, but JS callbacks (HTTP
> handlers, timers, etc.) queue up and don't fire until generate
> returns. For server processes that need to handle other requests
> during inference, run generate inside a `worker_threads` Worker.
> For one-off scripts the block is fine.

### Cancellation

`session.cancel()` flips an atomic that the decode loop checks at
every flush boundary; generation exits with `finishReason === "Cancelled"`
at the next checkpoint.

The catch: JS workers (web + `worker_threads`) are single-threaded.
While `generate()` is blocking, the worker's own message handlers
**cannot run** — a `postMessage({ kind: 'cancel' })` from the main
thread queues but doesn't dispatch until `generate` returns. So a
plain JS flag set by an `onmessage` handler can't be updated
mid-decode.

Two patterns that actually work:

**1. Cancel from inside the token callback** based on state the
callback can observe directly (elapsed time, accumulated content,
token budget beyond `opts.maxTokens`):

```js
const startMs = performance.now();
session.generate(opts, (toks) => {
    self.postMessage({ kind: 'tokens', toks });
    if (performance.now() - startMs > 30_000) {
        session.cancel();  // 30-second budget
    }
});
```

**2. `SharedArrayBuffer` + `Atomics`** for true cross-thread
signalling. Allocate an `Int32Array` on a `SharedArrayBuffer`,
poll it from inside the callback with `Atomics.load`, set it from
the main thread with `Atomics.store`. **Requires cross-origin
isolation** in browsers (`Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: require-corp` headers); transparent
on Node `worker_threads`.

```js
// main thread — only structured-cloneable data crosses the
// postMessage boundary. wasm-bindgen objects (GenerateOpts,
// Session, etc.) live inside wasm linear memory and would throw
// on postMessage; pass plain params and construct the object on
// the worker side.
const sab = new SharedArrayBuffer(4);
const cancelFlag = new Int32Array(sab);
worker.postMessage({ kind: 'init', cancelFlag });
worker.postMessage({ kind: 'generate', params: { maxTokens: 64 } });
// later, to cancel:
Atomics.store(cancelFlag, 0, 1);

// worker.js — `generate()` is invoked from inside the message
// handler so `cancelFlag` is guaranteed initialized first; running
// `generate` at top-level would race with `init` and crash on
// `Atomics.load(undefined, 0)`. The `cancelFlag` null check guards
// against a `generate` message arriving before `init` (out-of-order
// messages).
let cancelFlag;
self.onmessage = (ev) => {
    if (ev.data.kind === 'init') {
        cancelFlag = ev.data.cancelFlag;
    } else if (ev.data.kind === 'generate') {
        const opts = new GenerateOpts();
        Object.assign(opts, ev.data.params);  // setters fire per field
        session.generate(opts, (toks) => {
            self.postMessage({ kind: 'tokens', toks });
            if (cancelFlag && Atomics.load(cancelFlag, 0) !== 0) {
                session.cancel();
            }
        });
    }
};
```

A pure `postMessage`-flag pattern only works if the cancel arrives
before `generate` is called (the listener runs during the gap
between message receipt and the next `generate`). It's unreliable
for in-flight cancellation; use one of the two patterns above.

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

# wick-wasm

`wasm-bindgen` browser / Node bindings for the
[wick](https://github.com/hyeons-lab/wick) inference engine.

> Status: pre-1.0 skeleton. The current surface is intentionally small
> (`wickVersion()` + `Manifest` parse). Full engine, session, and
> tokenizer wrappers land in subsequent releases.

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

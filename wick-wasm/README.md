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

```js
import init, { wickVersion, Manifest } from '@hyeonslab/wick-wasm';

await init();  // loads the .wasm

console.log(wickVersion());  // e.g. "0.1.0"

const res = await fetch('/path/to/manifest.json');
const bytes = new Uint8Array(await res.arrayBuffer());
const manifest = Manifest.parse(bytes);

console.log(manifest.inferenceType);   // "llama.cpp/text-to-text"
console.log(manifest.modelUrl);        // "https://.../model.gguf"
console.log(manifest.schemaVersion);   // "1.0.0"
```

Required init pattern is bundler-specific; see the
[`wasm-pack` book](https://rustwasm.github.io/docs/wasm-pack/)
for webpack / Vite / Rollup loaders.

## Building from source

```sh
just wasm  # produces wick-wasm/pkg/
```

Requires `wasm-pack` (`cargo install wasm-pack`) and `wasm-opt`
(macOS: `brew install binaryen`; linux: `apt-get install binaryen`).

## License

Apache-2.0 OR MIT, matching the rest of the wick workspace.

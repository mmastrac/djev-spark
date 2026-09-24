# djev-spark

DiffusionGemma 26B-A4B (NVFP4) on a DGX Spark, serving structured decisions
through Jev's API.

One container runs vLLM on port 8010 and the structured decision server on
port 8011 in front of it. The server is [mmastrac/djev](https://github.com/mmastrac/djev);
it implements Jev's `POST /v1/systemone` API and passes ordinary chat through
to vLLM.

Structured reads are upstream in vLLM
([vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250)).
This image adds seven perf branches that are not upstream yet; see
[docs/vllm-stack.md](docs/vllm-stack.md).

## Image

| piece | from | pinned by |
|---|---|---|
| vLLM | `vllm/vllm-openai:nightly-e9757321`, vLLM 0.29.1rc1.dev573 | `BASE`, `VLLM_BASE` |
| perf overlay (12 files under `vllm/`) | fork branch `djev-spark-stack`: the nightly's commit plus the branches in [`vllm-stack.tsv`](vllm-stack.tsv) | `VLLM_REF` |
| structured server `/opt/dgemma/structured_server.py` | [mmastrac/djev](https://github.com/mmastrac/djev) | `DJEV_REF` |
| test pages `/opt/dgemma/pages/` | `pages/` in this repo | |
| `link_cuda_headers.sh` | this repo | links the CUDA headers and unversioned `.so` names the base leaves out, for FlashInfer's JIT |
| `worker_memory_cap.py`, `spark_mem_trace.py` | this repo | a per-worker memory cap, armed only by `TORCH_MEM_FRACTION` |

`patches/overlay_vllm.py` stops the build if the base's vLLM is not the
stack's commit. `patches/verify_image.py` stops it if any branch's change is
missing from the installed vLLM, a changed module does not import, or the
server lacks a flag the entrypoint passes.

`scripts/vllm-stack.sh status` reports, per branch, whether the fork moved
and whether upstream merged the PR into the base nightly yet, which is when
the branch leaves the stack.

## Requirements

It may work on other hardware. Reports are welcome.

- DGX Spark or other GB10 box (aarch64, unified memory, CUDA 13 driver).
- Docker with the NVIDIA runtime and BuildKit.
- 25 GB disk for the image, 18 GB for the checkpoint.
- Memory: weights 19 GB, KV pool `KV_CACHE_GB`, plus a start-up transient
  that scales with `MAX_SEQS x CANVAS` (`TRANSIENT_COPIES` sets the budget
  for it).

## Run

```bash
cp .env.example .env
scripts/download-model.sh       # nvidia/diffusiongemma-26B-A4B-it-NVFP4 -> $MODELS_DIR/$MODEL_NAME
docker compose up -d --build
docker compose logs -f dgemma
scripts/smoke.sh
```

`POST /v1/systemone` takes Jev's request body and returns Jev's answer
shapes. No API key unless `API_KEY` is set. The server ignores `model`.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {"type": "choice", "instructions": "Which team owns this?",
             "criteria": {"billing": null, "outage": "service down", "feature": null}},
    "tone": {"type": "score", "instructions": "How angry is the customer?",
             "criteria": ["calm", "annoyed", "furious"]}
  }}'
```

### Request

| field | description |
|---|---|
| `state` | The content the questions are asked about. A string, object or array. |
| `questions` | Map of question id to question object. Answers use the same ids, in the same order. |
| `model` | Accepted for compatibility with Jev clients and ignored. |
| `seed` | Optional. Seeds the noise draws, so the same request with the same seed gives the same answer. Default 42. |

Each question has a `type`, an `instructions` string (the question itself), and `criteria`, whose shape depends on the type:

| type | what it asks | `criteria` |
|---|---|---|
| `noul` | Yes or no. | Optional. `{"true": "what yes means", "false": "what no means"}`. The prompt shows the descriptions next to the labels. |
| `choice` | One of several options. | Required. Object mapping each option name to a description, or `null` for none. Option order is kept. |
| `score` | A level on an ordered scale. | Required. List of level names from lowest to highest, at least two. |

### Response

| field | what it holds |
|---|---|
| `answers` | One answer per question id. The shape depends on the question type (below). |
| `usage` | `input_tokens`: the prompt input count, images included. `output_tokens`: the canvas rows read, plus any thought tokens. |
| `diagnostics` | Server-specific, subject to change. |
| `model` | The served model name. |

| answer type | fields |
|---|---|
| `noul` | `noul`: probability of yes, 0 to 1. |
| `choice` | `choice`: the most probable option. `probabilities`: probability per option name. `confidence`: the probability of `choice`. |
| `score` | `score`: probability-weighted level index, lowest level 0. `legend`: level index to level name. `probabilities`: probability per level index. `confidence`: the probability of the most probable level. |

Validation errors return 422 with `{"error": {"message": ...}}`. A failed upstream call returns 502.

### Extensions

These keys extend Jev's API. They go at the top level of the request body,
beside `state` and `questions`.

| key | default | description |
|---|---|---|
| `samples` | `"auto"` | Number of noise draws to average. `"auto"` reads once, then more only if the first read's entropy is above `auto_threshold`. |
| `auto_max` | 4 | Maximum reads under `"auto"`. |
| `auto_threshold` | 0.1 | Entropy above which `"auto"` reads again. |
| `think` | 0 | The model writes up to this many tokens of thought before the read, and the read conditions on it. One extra generation per decision. With images the thought is seeded into the canvas, so the canvas bounds it. |
| `instructions` | | Context rendered ahead of the questions, so requests that share it share a KV prefix. |
| `chunk_rows` | canvas | Splits a long question list into chunks of at most this many rows. |
| `chunk_prompt` | `"own"` | Whether each chunk's prompt lists only its own questions (`"own"`) or every question (`"shared"`). |
| `sequential` | false | Runs chunks in order, prefilling each chunk's answers before the next, so later answers condition on earlier ones. Text-only states. |
| `ask` | | List of question ids to answer in one read. The rest are skipped. |
| `steps` | 1 | Denoise steps per read. More than 1 lets the canvas drift from the template. |

### Dependencies

By default the server reads every question in a request in one canvas. The
answers share the prompt and the canvas. No answer sees another. These
per-question keys change that.

| key | what it does |
|---|---|
| `depends_on` | List of question ids. This question is read in a later stage than those, with their answers in its prompt. |
| `ask_if` | Map of question id to a list of that question's answers (option names, level names, or `yes`/`no`). The server asks the question only when the answer is in the list. Otherwise its answer is `null`. Implies `depends_on`. |
| `alone` | `true` reads the question in a canvas of its own, beside the others in its stage. For questions that bias each other, like a direction question next to a hazard question. |

Questions run in stages by their dependencies, in declaration order within
a stage. A stage is one joint read, chunked by the canvas, with `alone`
questions in their own reads, all concurrent. Later stages continue the
earlier answers. For a text state that is a prefilled continuation of the
same prompt, about one read's cost per stage. For an image state it is a
fresh read with the earlier answers restated in the state text, one image
decision per stage. Cycles, unknown ids, `ask_if` values that are not
answers of the named question, and an `ask` list missing a dependency
are 422s. `diagnostics.stages` lists the ids read in each stage,
`diagnostics.skipped` the gated ones with the answer that gated them, and
`diagnostics.conditioning` is `prefill`, `restated` or null.

```json
"questions": {
  "ahead": {"type": "choice", "instructions": "...", "criteria": {"all clear ahead": null, "danger: wall ahead": null}},
  "side": {"type": "choice", "instructions": "Which half of the frame is the obstacle in?",
           "criteria": {"left half": null, "right half": null},
           "ask_if": {"ahead": ["danger: wall ahead"]}}
}
```

### Images

Jev's API has no images. This server takes them in either form below and
puts them ahead of the state in the prompt.

| form | how |
|---|---|
| multipart | `multipart/form-data` with the JSON body in a part named `request` and each image as a file part. Any part name, any number of images, in order. This is what `curl -F` and browser `FormData` send. |
| JSON | An `images` array in the body, each entry a `data:image/...;base64,...` URL or an object `{"content_type": "image/png", "base64": "..."}`. |

`sequential` needs a text-only state and returns 422 with images.

```bash
curl -s localhost:8011/v1/systemone \
  -F 'request={"model": "jev-latest", "state": {"note": "the photo is from the returns desk"},
               "questions": {"damaged": {"type": "noul", "instructions": "Is the item damaged?"}}}' \
  -F 'photo=@returns/1234.jpg'
```

Playground: with `TEST_PAGE=1`, `http://<box>:8011/` serves a page with the
request JSON in a textbox and an image mode dropdown: none, image file,
webcam one-shot (capture and send), webcam live (a frame every N seconds),
webcam realtime (capture again as each answer returns). `#req=<base64 JSON>`
in the URL fills and sends a request on load. `TEST_PAGE` is off by
default.

Webcam realtime mode:

![playground in webcam realtime mode](docs/playground.jpg)

Static image tests:

![playground with image](docs/triangle.png)

### From another device

The container is on the host network and both servers listen on all
interfaces, so `http://<box-ip>:8011/` works from the LAN with no port
mapping. Browsers open a webcam only on a secure origin, so for the
playground's webcam modes from another device set `TLS_PORT` (for example
8443) and use `https://<box-ip>:8443/`. The certificate is self-signed, so
the browser asks once.

### Walking demo

`https://<box-ip>:8443/walk` (with `TEST_PAGE=1` and `TLS_PORT=8443`) is a
phone page. It streams the back camera and sends a 512px frame per round
trip as one hazard question. It shows the label full width under the
video, a direction arrow over it and the probabilities below, with no
scrolling.

The hazard labels are all clear ahead, danger: stairs, danger: wall ahead,
danger: object ahead, danger: pet ahead and door ahead. The instructions
tell the model to judge only a 30 degree window at the center of the frame,
and to say all clear when nothing is within 2 meters there. A second
question, gated with `ask_if` on the danger labels, asks which half of the
frame the obstacle is in, and the arrow points the other way. On a clear
frame the second question is skipped and the arrow is go ahead.

Tests on frames with an obstacle on a known side decided that shape. Asking
for the turn directly ("which side is more open", turn left or turn right)
was biased right: p(turn left) was 0.03 to 0.25 whichever side the obstacle
was on, and listing right first flipped the bias. Asking where the obstacle
is was right at 0.95 to 1.00. The two questions also run in separate reads,
because a direction question beside the hazard question biased the hazard
slot toward obstacles (all clear on a clear frame fell from 0.8 to 0.04).

A clear frame is one image decision, about 320 ms on the Spark plus the
upload. A blocked frame is two. The "think" box adds a 64-token thought per
frame, written with the frame in view and seeded into the read's canvas
ahead of the answer. The canvas bounds the thought, so at 128 rows a
request for more is clipped and `diagnostics.thought.budget` says to what.

To reload the server or the pages without restarting vLLM, copy the files
into the container and kill the server process. The entrypoint restarts it.

```bash
docker cp ../djev/structured_server.py dgemma:/opt/dgemma/
docker cp pages/. dgemma:/opt/dgemma/pages/
docker exec dgemma pkill -f structured_server.py
```

### Cube Rule demo

`http://<box-ip>:8011/cube` (with `TEST_PAGE=1`) classifies 27 Wikipedia
food photos under the Cube Rule, one read per photo: toast, sandwich,
taco, sushi, quiche, calzone or salad by where the starch sits, plus soup
for a starchless food that is liquid in a bowl or a cup. The photos are
embedded in the page. Each one goes to the server this page came from, and
its card moves from the pool into the row of the category it chose, sorted
by confidence. Replay animates a recorded run without the server. One photo
is about a third of a second on the Spark.

### Other routes

| route | what it does |
|---|---|
| `POST /v1/chat/completions` | A decision when the system message is a JSON object with `questions`: the user message is the state JSON or image parts, and the reply `content` is the answer JSON. The schema is documented at the top of djev's `structured_server.py`. Any other chat goes to vLLM unchanged, streaming and tool calls included. |
| `POST /v1/raw/chat/completions` | Always passes the body to vLLM's chat completions unchanged: plain generation through the same port and, with `API_KEY`, the same token. |
| other `POST /v1/...` | Passed to vLLM, for example `/v1/completions`. |
| `GET /v1/models` | vLLM's model list; 503 until vLLM is up. |
| `GET /health` | Always open, no token. |

## Configuration

Environment variables, same defaults in `compose.yaml` and `.env.example`.

| variable | default | meaning |
|---|---|---|
| `MODELS_DIR`, `MODEL_NAME` | `./models`, `dgemma` | checkpoint at `$MODELS_DIR/$MODEL_NAME` |
| `CACHE_DIR` | `./cache` | flashinfer autotune and torch compile cache |
| `CANVAS` | 256 | served canvas in tokens. A read pays only for its own width |
| `CANVAS_SCHEDULE` | `[[1, 2, 256], [3, 6, 128], [7, 32, 64]]` | generation block width by load, `[low, high, width]` over running plus waiting requests. No width may exceed `CANVAS`; empty uses `CANVAS` for every block |
| `MAX_SEQS` | 32 | concurrent requests |
| `MAX_MODEL_LEN` | 4096 | prompt plus canvas |
| `GPU_UTIL` | 0.40 | fraction of box memory vLLM plans for |
| `KV_CACHE_GB` | 2 | KV pool, fixed |
| `MAX_NUM_BATCHED_TOKENS` | empty | prefill chunk (empty = vLLM default) |
| `ATTN` | TRITON_ATTN | attention backend |
| `EXTRA_ARGS` | empty | appended to `vllm serve` |
| `CONSTRAINED` | 1 | reads over the labels only (`--constrained`) |
| `ENGINE_SAMPLES` | 1 | a fixed sample count as one `diffusion_samples` request (`--engine-samples`) |
| `MAX_SAMPLES` | 32 | cap on samples per question, in the server and the engine |
| `REASONING_PARSER` | gemma4 | strips the empty thought block from plain chat; empty turns it off |
| `TOOL_CALL_PARSER` | gemma4 | tool calls in every `tool_choice` mode; empty turns them off |
| `HEADROOM_GB` | 12 | free memory required beyond weights, KV and transient |
| `TRANSIENT_COPIES` | 2 | fp32 `[MAX_SEQS x CANVAS, vocab]` copies the start-up budget allows for |
| `TORCH_MEM_FRACTION` | empty | per-worker cap (empty = unbounded) |
| `TEST_PAGE` | empty | `1` serves the playground at `/`, and `/walk` and `/cube`, on the structured port |
| `TLS_PORT` | 0 | nonzero adds an HTTPS listener with a self-signed certificate. Browsers need it to open a webcam from another device |
| `API_KEY` | empty | when set, POST routes on the structured port need `Authorization: Bearer <key>` |
| `PORT`, `STRUCTURED_PORT` | 8010, 8011 | host network |

128k profile (in `.env.example`, commented): `MAX_MODEL_LEN=131072
KV_CACHE_GB=24 GPU_UTIL=0.45`.

## Benchmarks

One GX10 (GB10, 121 GB), 2026-09-24, with nothing else running on the box.
Defaults: `CANVAS=256` with the adaptive schedule, `MAX_SEQS=32`,
`MAX_MODEL_LEN=4096`, `KV_CACHE_GB=2` (18,995 tokens), `TRITON_ATTN`.

Start-up with a warm cache: 172 s to serving, 136 s of it loading weights and
11 s engine init. The container used at most 29 GB of host memory.

Structured reads, `scripts/read-curve.py`: a unique three-question state
(yes/no, choice, score) per request, 15 s per point.

| samples | clients | req/s | decisions/s | p50 s | p95 s |
|---|---|---|---|---|---|
| 1 | 1 | 9.58 | 28.7 | 0.10 | 0.11 |
| 1 | 8 | 37.45 | 112.3 | 0.21 | 0.23 |
| 1 | 16 | 60.72 | 182.2 | 0.26 | 0.29 |
| 1 | 32 | 87.74 | 263.2 | 0.36 | 0.45 |
| 4 | 1 | 7.86 | 23.6 | 0.12 | 0.17 |
| 4 | 32 | 28.16 | 84.5 | 1.13 | 1.15 |

Plain generation, `scripts/gen-bench.py`: 200-token completions (every
request ran to the cap), completion tokens per second of wall time.

| concurrent | tok/s | req/s | mean s per request |
|---|---|---|---|
| 1 | 94.4 | 0.47 | 2.12 |
| 4 | 144.4 | 0.72 | 4.73 |
| 8 | 210.3 | 1.05 | 6.84 |
| 16 | 261.2 | 1.31 | 10.63 |
| 32 | 302.5 | 1.51 | 18.26 |

The first request at a new canvas width or batch size compiles once; both
scripts warm up before timing.

Two changes that looked like headroom, measured the same way, one run each:

| | reads, 1 sample, 32 clients | reads, 4 samples, 32 clients | generation, 1 / 8 / 32 concurrent | start-up cost |
|---|---|---|---|---|
| defaults | 87.7 req/s | 28.2 req/s | 94 / 210 / 302 tok/s | |
| CUDA graphs captured to 2048 tokens (default 512) | 85.2 | 28.9 | 87 / 212 / 314 | +20 s capture, +1 GiB |
| `ATTN=FLASHINFER` | 85.3 | 28.2 | 96 / 208 / 290 | +90 s engine init |

Neither is outside run-to-run noise, so the defaults stay. FlashInfer works
here only because of the `flashinfer-per-request-causal` branch.

Long states, 128k profile (`MAX_MODEL_LEN=131072 KV_CACHE_GB=24
GPU_UTIL=0.45`), `scripts/long-context-probe.py`: one two-question decision
per state, cold and then warm (prefix cached). The KV pool holds 1,808,085
tokens, 13.79 requests of 128k: 25 of the 30 layers are sliding-window, and
the allocator gives them only the window.

| state tokens | cold s | warm s |
|---|---|---|
| 8,678 | 2.61 | 0.13 |
| 35,133 | 13.63 | 0.37 |
| 121,927 | 131.50 | 0.50 |

## Tests

- The build runs `patches/verify_image.py`: every perf branch present, the
  changed modules import, the server has the flags the entrypoint passes.
- `scripts/smoke.sh`: one generation on 8010, one decision on 8011.
- `scripts/long-context-probe.py [tokens ...]`: cold and warm decision
  latency over states of the given sizes.
- `scripts/read-curve.py`, `scripts/gen-bench.py`: the benchmarks above.

## Files

```
Dockerfile                      base + perf overlay + djev server + patches
vllm-stack.tsv                  the perf branches, pinned heads and upstream PRs
compose.yaml                    one service, host network
entrypoint.sh                   memory guard, vllm serve, structured server
.env.example
docs/vllm-stack.md              the stack: what each branch does, how to keep it current
patches/link_cuda_headers.sh
patches/overlay_vllm.py
patches/verify_image.py
patches/worker_memory_cap.py
patches/spark_mem_trace.py
pages/index.html                the playground
pages/walk.html
pages/cube.html
scripts/vllm-stack.sh           status of the perf branches; build a new stack
scripts/download-model.sh
scripts/smoke.sh
scripts/long-context-probe.py
scripts/read-curve.py
scripts/gen-bench.py
```

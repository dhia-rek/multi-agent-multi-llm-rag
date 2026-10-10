# Deploying to Hugging Face Spaces

The repo ships everything a Docker Space needs: `Dockerfile`, `deploy/start.sh`
(builds the FAISS index on first boot, then starts Streamlit on port 7860) and a
Space card (`deploy/huggingface/README.md`).

## Steps

1. Create a Space at https://huggingface.co/new-space — SDK **Docker**, hardware
   **CPU basic** (free) is enough.
2. Push this repo to the Space, replacing the root `README.md` with the Space card
   (Spaces read their config from the README front matter):

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space> space
   git archive HEAD | tar -x -C space
   cp deploy/huggingface/README.md space/README.md
   cd space
   git lfs install && git lfs track "*.pdf"   # the Hub rejects binary files outside LFS/Xet
   git add -A && git commit -m "Deploy" && git push
   ```
3. Optional — live models: Space **Settings → Variables and secrets → New secret**
   `GOOGLE_API_KEY`. Without it the Space stays in demo mode.

## Secrets and variables

| Name | Kind | Default | Effect |
|---|---|---|---|
| `GOOGLE_API_KEY` | secret | unset | Enables live Gemini calls. Public visitors spend your free-tier quota. |
| `DEMO_MODE` | variable | `auto` | `auto`: demo when no key · `on`: always demo · `off`: never |
| `GEMINI_FAST_MODEL` / `GEMINI_POWERFUL_MODEL` | variable | `gemini-2.5-flash-lite` / `gemini-2.5-flash` | Model per routing tier |

First boot takes ~1–2 minutes (image build, embedding model download, index build).
The index lives in the container and is rebuilt after each restart.

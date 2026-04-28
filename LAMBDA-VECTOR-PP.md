# lambda-vector-pp branch

A long-running fork of `kvcache-ai/sglang` (which is itself a fork of
`sgl-project/sglang` with kt-kernel CPU expert offload integration) carrying
the local patches needed to run heterogeneous pipeline-parallel deployments
on the Lambda Vector box: an RTX 3060 (sm_86, 12 GB) on PP rank 0 and an
RTX 4090 (sm_89, 24 GB) on PP rank 1, with kt-kernel CPU offload pinned to
rank 1 only.

## Why this branch exists

Upstream `kvcache-ai/sglang` doesn't support:

  * Per-PP-rank model paths (we need AWQ-Int4 on the 3060 + FP8 on the 4090
    of the same model — sm_86 can't run FP8 MoE GEMMs).
  * Several Qwen3.5/Qwen3-Next + AWQ + PP loading bugs that surface only
    under our heterogeneous-quant configuration.
  * Mamba-state release in the dynamic-chunking profile loop (real upstream
    bug, candidate for upstreaming).
  * Source-built sgl-kernel that registers under the `sglang-kernel`
    distribution name rather than the PyPI-published `sgl-kernel`.

Each commit on this branch fixes one of those gaps with a minimal, scoped
change and a commit message that explains the bug and the reasoning. See
`git log lambda-vector-pp` for the full list.

## Prebuilt images

Every push to `lambda-vector-pp` triggers `.github/workflows/build-image.yml`
which builds a Docker image for **sm_86 + sm_89 only** (no Hopper, no
Blackwell — saves ~80% of build time vs the upstream multi-arch matrix) and
pushes it to GitHub Container Registry.

Pull the latest:

```bash
docker pull ghcr.io/empty-quiver/sglang-kt:lambda-vector-pp
```

Pin to a specific commit (recommended for production):

```bash
docker pull ghcr.io/empty-quiver/sglang-kt:sha-<12-char-prefix>
```

## How to add a new patch

1. Create your change as a single logical commit on the `lambda-vector-pp`
   branch. Commit message format:
   ```
   <area>: <one-line summary under 70 chars>
   
   <2-3 paragraphs explaining the bug/feature, the fix, and any non-obvious
   correctness reasoning. Cite vector-llm-stacks commit SHAs where relevant.>
   ```

2. **Don't touch tests.** If a patch accidentally modifies a test file,
   that's a bug in the patch — fix the patch, don't disable the test.

3. Push to GitHub:
   ```bash
   git push origin lambda-vector-pp
   ```

4. The GHA workflow runs automatically (~30-90 minutes for a clean build,
   ~10-30 minutes when buildx GHA cache is warm). Monitor at
   `https://github.com/empty-quiver/sglang/actions`.

5. Once the workflow turns green, update the `image:` field in
   `vector-llm-stacks/docker-compose.kt-pp.yml` to point at the new
   `sha-<N>` tag, run `docker compose up -d`, and verify boot + a smoke
   inference before declaring the upgrade done.

## How to rebase against upstream

```bash
git fetch upstream
git rebase upstream/main
# Resolve conflicts (rare; most patches are scoped to non-conflicting hotspots)
git push --force-with-lease origin lambda-vector-pp
```

The GHA workflow will rebuild the image. If a rebase touches Dockerfile
fundamentals, also test the boot locally before pushing.

## Running locally without GHA

If you need to build without pushing (e.g., for a hotfix that hasn't been
committed yet):

```bash
cd /home/eve/sglang-kt-fork
docker build -f docker/lambda-vector-pp.Dockerfile -t sglang-kt:local . \
    --progress=plain
```

The Dockerfile uses BuildKit cache mounts (`--mount=type=cache,...`) for
ccache and uv, so subsequent local builds are fast.

## Layout

```
.github/workflows/build-image.yml   # GHA workflow (sm_86+sm_89, push to ghcr)
docker/lambda-vector-pp.Dockerfile   # Build recipe
docker/kt-kernel-numa-single-socket.patch
                                    # Community patch for ktransformers#1754
                                    # (single-socket NUMA segfault fix). Lives
                                    # here because it's against kvcache-ai/
                                    # ktransformers (kt-kernel), not sglang.
LAMBDA-VECTOR-PP.md                 # This file.
```

## Don't push to upstream

`upstream` remote points at `kvcache-ai/sglang` (which we don't own).
Pushing there is a hard error — `git push origin <branch>` is what you
want, where `origin` is `empty-quiver/sglang`.

If a patch on this branch is genuinely upstream-quality (e.g., the Mamba
leak fix), open a separate PR against `sgl-project/sglang` rather than
`kvcache-ai/sglang` — kvcache-ai's fork periodically syncs from the OSS
repo.

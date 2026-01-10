## Purpose

This repository is a small CLI utility for audio tone analysis. The authoritative implementation lives in `audio_stats.py`. These instructions are meant to help an AI coding agent be productive quickly by describing the big-picture flow, important patterns, how to run the project locally, and a few discovered gotchas.

## Big picture (what to know first)

- Single-file CLI tool: most logic is in `audio_stats.py`. Key classes:
  - `AudioAnalyzer` — core analysis pipeline (load/generate audio → window → FFT → component extraction → compute stats).
  - `AudioStats` — container + pretty-print helpers for results.
  - `AudioComp` — small struct for component frequency + power.
- Data flow: load (or generate) audio → `get_freq`/truncate → `get_bins` (window + rFFT) → `get_comps` (iterative peak extraction, zero bins) → `analyze` (classify single/two-tone, compute SFDR/SNR/THD/etc) → output via `print` / table.
- Why this structure: script-style CLI for quick measurements; numeric code uses numpy/scipy/librosa for signal ops and librosa for file I/O and tone generation.

## How to run (Windows PowerShell examples)

- Install minimal dependencies (PowerShell):

```powershell
python -m pip install numpy scipy librosa
```

- Generate test tones and show the table:

```powershell
python .\audio_stats.py -g -t
```

- Analyze files matching a glob, print components and stats:

```powershell
python .\audio_stats.py "samples\*.wav" -c -s -t
```

Notes: `-g` = generate, `-c` = components, `-s` = statistics, `-t` = table.

## Project-specific conventions & patterns

- CLI-first: add or change flags in `main()` (argparse) and ensure downstream `print` / `print_summary` / `print_header` keep formatting consistent.
- Stats pairings: when adding a new metric, update both `AudioStats.stats()` (dict) and `AudioStats.stats_index()` to preserve presentation ordering.
- FFT / bins conventions:
  - `get_bins()` applies a Blackman–Harris window then `rfft` and stores squared magnitude in `self.bins`.
  - `get_comps()` mutates `self.bins` by zeroing out ranges around found peaks — treat `self.bins` as single-use between `get_bins()` and `get_comps()`.
  - Frequency mapping helpers: `bin_to_hz(bin)` and `freq_to_bin(freq)` are the canonical conversions — use them to keep unit consistency.
- Component extraction: `get_comps()` extracts N peaks by repeatedly finding `bins.max()` and zeroing ±w_tol around it; the returned `self.comps` are in descending power order.

## Integration points & dependencies

- External libs: `numpy`, `scipy` (signal, fftpack), `librosa` (I/O & tone generator). Changes to signal processing must respect these APIs.
- I/O: file load uses `librosa.load(fname, sr=None)` — it preserves original sample rate; downstream code expects `self.fs` to be set accordingly.

## Important implementation notes & gotchas (discovered while reading the code)

- Two-tone detection check looks suspicious: in `analyze()` the condition
  `abs(self.comps[1].freq - 2*self.comps[0].pwr) < 5` appears to compare frequency against power (likely a typo). When modifying two-tone logic, verify this carefully.
- `get_freq()` uses zero-crossings to infer fundamental and truncates the audio to the first/last crossing — this shortens the buffer and changes subsequent FFT resolution. Tests or edits that assume full-length buffers will fail unless handled.
- `db()` uses `librosa.power_to_db(..., amin=1e-12)`: expect large negative dB values instead of -inf for tiny powers.
- `get_comps()` sets `self.dc` to sum of low-frequency bins then zeroes them. This mutates `self.bins`; if you need original bins keep a copy.

## Editing guidance (how to change code safely)

- When updating numeric/algorithmic code, add a small deterministic unit test that uses `AudioAnalyzer.gen_test_audio()` (existing generator) so you can assert expected dB differences and THD%. Put tests under `tests/` and keep them small and deterministic.
- If you add new dependencies, also add a `requirements.txt` and update this README and CI as needed.
- Preserve the CLI behavior and `AudioStats` formatting unless intentionally changing output contracts.

## Key files to inspect for future changes

- `audio_stats.py` — main implementation and CLI (single source of truth for behavior/formatting). 

---

If anything here is unclear or you want the instructions expanded (CI snippets, unit-test templates, or a requirements file), tell me which section to expand and I will update the document.

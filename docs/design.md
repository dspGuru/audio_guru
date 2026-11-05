# Audio Guru Design Document

This document describes the architectural design and code structure of the Audio Guru library. For detailed explanations of the signal processing algorithms and metric calculations, see [overview.md](overview.md).

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Organization](#module-organization)
3. [Core Classes](#core-classes)
4. [Data Flow](#data-flow)
5. [Class Hierarchies](#class-hierarchies)
6. [Design Patterns](#design-patterns)
7. [Extension Points](#extension-points)

---

## Architecture Overview

Audio Guru follows a layered architecture separating concerns into distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI / Entry Points                         │
│                    analyze.py, generate.py                      │
├─────────────────────────────────────────────────────────────────┤
│                     Orchestration Layer                         │
│                      AudioAnalyzer                              │
├─────────────────────────────────────────────────────────────────┤
│                   Category Analyzers                            │
│     ToneAnalyzer │ NoiseAnalyzer │ SweepAnalyzer │ TimeAnalyzer │
├─────────────────────────────────────────────────────────────────┤
│                    Analysis Results                             │
│      ToneStats  │  FreqResp  │  TimeStats  │  Components       │
├─────────────────────────────────────────────────────────────────┤
│                   Signal Processing                             │
│              FreqBins │ Component │ Segment                     │
├─────────────────────────────────────────────────────────────────┤
│                      Core Data                                  │
│                Audio │ Metadata │ Segment                       │
├─────────────────────────────────────────────────────────────────┤
│                    External I/O                                 │
│               soundfile │ sounddevice │ numpy                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Single Responsibility**: Each class handles one aspect of analysis
- **Composition over Inheritance**: Analyzers compose Audio and FreqBins objects
- **Lazy Evaluation**: FFT bins computed on first access, cached thereafter
- **Immutable Results**: Analysis objects are created once, not modified

---

## Module Organization

### Entry Points

| Module | Purpose |
|--------|---------|
| `analyze.py` | CLI for analyzing audio files/devices |
| `generate.py` | Signal generation utilities and CLI |
| `play.py` | CLI for playing audio files |
| `split.py` | CLI for splitting audio files into segments |
| `join.py` | CLI for joining audio files with optional silence |
| `device_selector.py` | CLI for listing and selecting audio devices |

### Core Data

| Module | Purpose |
|--------|---------|
| `audio.py` | Audio container, I/O, segmentation |
| `metadata.py` | File/device metadata container |
| `segment.py` | Time region descriptors |
| `constants.py` | Library-wide constants and defaults |
| `util.py` | Enums (Category, Channel) and helpers |
| `audio_list.py` | Collection of Audio objects for batch I/O |
| `audio_player.py` | Audio playback wrapper around sounddevice |
| `audio_select.py` | Device enumeration, categorization, and sorting |

### Signal Processing

| Module | Purpose |
|--------|---------|
| `freq_bins.py` | FFT, windowing, component extraction |
| `component.py` | Frequency component representation |
| `decibels.py` | dB conversion utilities |

### Analysis

| Module | Purpose |
|--------|---------|
| `audio_analyzer.py` | Orchestrates multi-segment analysis |
| `analyzer.py` | Abstract base for category analyzers |
| `tone_analyzer.py` | Single/two-tone analysis |
| `noise_analyzer.py` | Noise spectrum analysis |
| `sweep_analyzer.py` | Frequency sweep analysis |
| `time_analyzer.py` | Time-domain only analysis |

### Results

| Module | Purpose |
|--------|---------|
| `analysis.py` | Abstract base for results |
| `tone_stats.py` | THD, SNR, SFDR metrics |
| `time_stats.py` | RMS, crest factor, phase metrics |
| `freq_resp.py` | Frequency response metrics |
| `audio_stats.py` | Aggregated statistics container |
| `unit_stats.py` | Device-under-test unit specification summary |

---

## Core Classes

### Audio (`audio.py`)

The central data container managing audio samples and their lifecycle.

```
Audio
├── samples: np.ndarray          # Raw sample buffer
├── segment: Segment             # Current selection bounds
├── md: Metadata                 # File/source metadata
├── bins: FreqBins (cached)      # FFT results (lazy)
│
├── I/O Methods
│   ├── open(fname) → context manager
│   ├── open_device(device) → context manager
│   ├── read(fname) → bool
│   ├── read_block() → bool
│   ├── write(fname) → bool
│   └── close()
│
├── Selection Methods
│   ├── select(segment)          # Set analysis region
│   ├── get_segments() → list    # Auto-detect segments
│   └── selected_samples → array # Windowed view
│
├── Properties
│   ├── dc, min, max, rms        # Time-domain stats
│   ├── freq                     # Estimated frequency
│   ├── cat → Category           # Auto-detected type
│   └── is_quadrature → bool     # Stereo phase check
│
└── Iterator Protocol
    └── __iter__, __next__       # Yields segment Audio objects
```

**Key Design Decisions:**
- Context manager pattern ensures proper resource cleanup
- `selected_samples` property provides windowed view without copying
- `bins` property uses lazy initialization (see [overview.md](overview.md))
- Iterator yields new Audio objects for each segment, preserving immutability

### FreqBins (`freq_bins.py`)

Encapsulates FFT computation and frequency-domain operations.

```
FreqBins
├── bins: np.ndarray             # Power spectral density
├── md: Metadata                 # Source metadata
├── channel: Channel             # Which channel(s) analyzed
│
├── Computed Properties
│   ├── tones → Components       # Peak detection (mutates bins)
│   ├── noise → float            # Residual after tone extraction
│   ├── max, min → float         # Bin extrema
│
├── Query Methods
│   ├── bin_to_freq(bin) → Hz
│   ├── freq_to_bin(Hz) → bin
│   ├── mean(f1, f2) → power
│   ├── ripple(f1, f2) → power
│   └── get_band_edges() → (low, high)
│
├── Detection Methods
│   ├── is_sine() → bool
│   └── is_noise() → bool
│
└── Extraction Methods
    ├── get_components(n) → Components
    └── get_bands(centers) → Components
```

**Important**: The `tones` property **mutates** a copy of the `bins` array by zeroing detected tone regions to compute `self.noise`. See [overview.md](overview.md) for algorithm details.

### Segment (`segment.py`)

Describes a contiguous region of audio with metadata.

```
Segment
├── fs: float                    # Sample rate
├── start: int                   # Start index (inclusive)
├── stop: int                    # Stop index (exclusive)
├── id: int | None               # Segment identifier
├── cat: Category                # Content category
├── complete: bool               # Fully analyzed flag
├── channel: Channel             # Channel assignment
│
├── Properties
│   ├── secs → duration
│   ├── start_secs, stop_secs
│   ├── slice → Python slice
│   └── desc, title → strings
│
└── Methods
    ├── limit(max_secs)          # Truncate length
    ├── offset(samples)          # Shift indices
    └── trunc(multiple)          # Round to seconds
```

**Design Note**: Uses Python slice semantics where `stop` is exclusive. The `Segments` container class provides `combine()` and `merge()` for managing segment lists.

### Component (`component.py`)

Represents a detected frequency component.

```
Component (dataclass)
├── freq: float                  # Center frequency (Hz)
├── pwr: float                   # Power level
├── secs: float                  # Timestamp
├── cat: Category                # Tone/Harmonic/Spurious/Noise/etc.
├── lower_freq: float            # Band lower edge
└── upper_freq: float            # Band upper edge

Components (list[Component])
├── md: Metadata                 # Source metadata
├── ref_pwr: float               # Reference for dB calculations
│
├── Manipulation
│   ├── combine(tol)             # Merge adjacent entries
│   ├── average() → Component    # Power-weighted mean
│   ├── sort(key)
│   └── normalize_pwr(ref_freq)
│
├── Query
│   ├── find_freq(Hz) → Component
│   ├── find_index(Hz) → int
│   ├── get_bands(centers) → Components
│   ├── band_edges(low, high) → (Hz, Hz)
│   └── ripple(f1, f2) → power
│
└── Export
    ├── get_dataframe() → pd.DataFrame
    └── to_csv(path)
```

---

## Data Flow

### File Analysis Flow

```
analyze.py
    │
    ▼
AudioAnalyzer.read(pattern)
    │
    ├── For each file:
    │       │
    │       ▼
    │   Audio.open(fname) ──────────────────┐
    │       │                               │
    │       ▼                               │
    │   Audio.read_block()                  │
    │       │                               │
    │       ▼                               │
    │   Audio.get_segments()                │
    │       │                               │  Context
    │       ▼                               │  Manager
    │   for segment_audio in audio:         │
    │       │                               │
    │       ▼                               │
    │   AudioAnalyzer.analyze(segment_audio)│
    │       │                               │
    │       ├── Detect category             │
    │       ├── Select analyzer             │
    │       ├── analyzer.analyze()          │
    │       └── Collect results             │
    │                                       │
    └───────────────────────────────────────┘
            │
            ▼
    AudioStats (aggregated results)
```

### Category Routing

`AudioAnalyzer` routes segments to specialized analyzers based on detected category:

```python
    analysis_classes = {
    Category.Noise: NoiseAnalyzer,
    Category.Silence: TimeAnalyzer,
    Category.Sweep: SweepAnalyzer,
    Category.Tone: ToneAnalyzer,
    Category.Unknown: TimeAnalyzer,
}
```

For details on how categories are detected, see [overview.md](overview.md).

### Analyzer Execution

Each analyzer follows the same pattern:

```
Analyzer.analyze()
    │
    ├── super().analyze()           # TimeStats computation
    │       │
    │       └── self.time_stats = TimeStats(audio)
    │
    ├── self.get_components()       # Category-specific extraction
    │       │
    │       └── audio.bins.tones / STFT / etc.
    │
    └── Create result object        # ToneStats / FreqResp / etc.
```

---

## Class Hierarchies

### Analyzer Hierarchy

```
Analyzer (ABC)
│
├── ToneAnalyzer
│   └── Produces: ToneStats, Components
│
├── NoiseAnalyzer
│   └── Produces: FreqResp, Components
│
├── SweepAnalyzer
│   └── Produces: FreqResp, Components
│
└── TimeAnalyzer
    └── Produces: TimeStats
```

All analyzers inherit from `Analyzer` which provides:
- `audio` attribute management
- `select()` for segment selection
- `get_dataframe()` / `to_csv()` for export
- Abstract methods: `analyze()`, `get_components()`, `reset()`

### Analysis Result Hierarchy

```
Analysis (ABC)
│
├── Properties from Metadata
│   ├── cat, fs, secs, segment
│   ├── start, unit_id
│
├── Abstract Methods
│   ├── to_dict() → dict
│   ├── summary() → str
│   └── summary_header() → str (static)
│
└── Concrete Classes
    ├── ToneStats      # THD, SNR, SFDR, THDN
    ├── TimeStats      # RMS, DC, crest, phase
    ├── FreqResp       # Ripple, band edges
    ├── Component      # Single frequency point
    └── UnitStats      # Aggregated unit specifications
```

For metric calculation details, see [Audio Performance Metrics](overview.md#audio-performance-metrics).

---

## Design Patterns

### Context Manager (Audio I/O)

```python
with Audio().open(fname) as audio:
    while audio.read_block():
        for segment in audio:
            process(segment)
# File automatically closed
```

**Benefits:**
- Guarantees resource cleanup
- Supports both file and device sources
- Enables streaming analysis of large files

### Lazy Initialization (FreqBins)

```python
@cached_property
def bins(self) -> FreqBins:
    return FreqBins(self.selected_samples, self.md, self.segment.channel)


```

**Benefits:**
- FFT computed only when needed
- Cached for repeated access
- Invalidated on segment change

### Factory Method (Analyzer Selection)

```python
def _get_analyzer(self, audio: Audio) -> Analyzer:
    cat = audio.cat
    analyzer_class = self.analysis_classes.get(cat, TimeAnalyzer)
    return analyzer_class(audio)
```

**Benefits:**
- Decouples category detection from analyzer instantiation
- Easy to add new categories/analyzers
- Testable with mock analyzers

### Iterator Protocol (Segment Processing)

```python
class Audio:
    def __iter__(self):
        if not getattr(self, "_segments_current", False):
            self.segments = self.get_segments()
        self._segments_current = False

        return self

    def __next__(self) -> "Audio":
        if not self.segments or not self.segments[0].complete:
            raise StopIteration
        segment = self.segments.pop(0)
        self.select(segment)
        return self.copy(self.segment)


```

**Benefits:**
- Natural Python iteration syntax
- Memory-efficient streaming
- Each iteration yields isolated Audio object

### Dataclass for Value Objects (Component)

```python
@dataclass
class Component(Analysis):
    freq: float = 0.0
    pwr: float = 0.0
    secs: float = 0.0
    cat: Category = Category.Unknown
    lower_freq: float = 0.0
    upper_freq: float = 0.0
```

**Benefits:**
- Automatic `__init__`, `__repr__`, `__eq__`
- Immutable-friendly (though not enforced)
- Clear documentation of attributes

---

## Extension Points

### Adding a New Analyzer

1. **Create analyzer class** inheriting from `Analyzer`:

```python
class MyAnalyzer(Analyzer):
    def reset(self) -> None:
        super().reset()
        # Initialize custom state

    def get_components(self) -> Components:
        # Extract frequency components
        return components

    def analyze(self) -> MyStats:
        super().analyze()
        self.components = self.get_components()
        return MyStats(self.components)
```

2. **Create result class** inheriting from `Analysis`:

```python
class MyStats(Analysis):
    def __init__(self, components: Components):
        super().__init__(components.md)
        # Calculate metrics

    def to_dict(self) -> dict:
        return {"metric1": self.metric1, ...}

    def summary(self) -> str:
        return f"{self.md.unit_id} {self.metric1}"

    @staticmethod
    def summary_header() -> str:
        return "Unit  Metric1  Metric2"
```

3. **Register in AudioAnalyzer**:

```python
analysis_classes[Category.MyCategory] = MyAnalyzer
```

### Adding a New Metric

To add a metric to an existing stats class (see [overview.md](overview.md) for context):

1. Add calculation in `__init__`
2. Add to `to_dict()` return value
3. Add to `summary()` output
4. Update `summary_header()` if displayed in tables

### Adding a New Signal Generator

In `generate.py`:

```python
def my_signal(
    freq: float = DEFAULT_FREQ,
    amp: float = DEFAULT_AMP,
    secs: float = DEFAULT_GEN_SECS,
    fs: float = DEFAULT_FS,
    channel: Channel = Channel.Stereo,
) -> Audio:
    # Validate inputs
    if freq <= 0:
        raise ValueError("Frequency must be positive")

    # Generate samples
    n_samples = round(secs * fs)
    t = np.arange(n_samples) / fs
    samples = amp * my_waveform(t, freq)

    # Package in Audio
    audio = Audio(fs=fs, channel=channel)
    audio.samples = samples.astype(Audio.DTYPE)
    audio.name = "MySignal"
    # ...
    return audio
```

### Custom Window Functions

To add a window function in `freq_bins.py`:

```python
MY_WINDOW_COEFFS = [...]  # Cosine series coefficients

def my_window(m: int, sym: bool = True) -> np.ndarray:
    return scipy.signal.windows.general_cosine(m, MY_WINDOW_COEFFS, sym=sym)
```

Then modify `FreqBins.__init__` to use the new window.

---

## Threading Considerations

`AudioAnalyzer` uses a worker thread for analysis:

```python
def _analyze_worker(self):
    while True:
        msg = self.msg_queue.get()
        if msg is None:
            break

        self.analyze(msg)
        self.msg_queue.task_done()




```

**Thread Safety:**
- Analysis queue is `queue.Queue` (thread-safe)
- Results appended to list under GIL protection
- No shared mutable state between segments

**Limitations:**
- Single worker thread (not parallel)
- Designed for I/O overlap, not CPU parallelism
- NumPy/SciPy release GIL during computation

---

## Memory Management

### Large File Handling

Audio files are processed in blocks (`DEFAULT_BLOCKSIZE_SECS = 5.0`):

```python
with Audio().open(large_file) as audio:
    while audio.read_block():  # Reads 5 seconds
        for segment in audio:
            analyze(segment)
        audio.samples = np.array([], dtype=audio.DTYPE)  # Release
```

### FFT Buffer Reuse

`FreqBins` uses `scipy.fft.rfft` which allocates output arrays. For repeated analysis, consider:

```python
# Clear cached bins to release memory
delattr(audio, "bins")
```

### Component List Growth

`Components` extends Python `list`. For very long recordings with many components, consider using `max_components` parameter:

```python
components = freq_bins.get_components(max_components=100)
```

---

## Configuration

### Constants (`constants.py`)

Key configurable values:

| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_FS` | 44100.0 | Sample rate (Hz) |
| `DEFAULT_BLOCKSIZE_SECS` | 5.0 | Streaming block size |
| `DEFAULT_MAX_ANALYSIS_SECS` | 30.0 | Max segment length |
| `SEGMENT_MIN_SECS` | 1.0 | Min analyzable segment |
| `MAX_TONES` | 40 | Peak detection limit |
| `MAX_BINS_PER_TONE` | 25 | Tone width in bins |
| `MIN_PWR` | 1e-12 | -120 dB floor |

### Per-Instance Configuration

```python
audio = Audio(
    fs=48000,                    # Override sample rate
    min_segment_secs=0.5,        # Shorter segments OK
    max_segment_secs=60.0,       # Longer segments OK
)
```

---

## Summary

Audio Guru's design emphasizes:

1. **Separation of Concerns**: Data (Audio), processing (FreqBins), analysis (Analyzers), and results (Stats) are cleanly separated

2. **Extensibility**: Abstract base classes and factory patterns make adding new analyzers straightforward

3. **Efficiency**: Lazy evaluation, caching, and streaming support large file analysis

4. **Pythonic API**: Context managers, iterators, and properties provide idiomatic interfaces

5. **Testability**: Dependency injection via composition enables mock-based testing

For implementation details of the signal processing algorithms, refer to [overview.md](overview.md).

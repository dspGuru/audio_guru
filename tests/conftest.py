from pathlib import Path
import sys

import pytest

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from generate import write_examples, write_signals


def cleanup_wav_files(directory: Path) -> None:
    """Delete all .wav files in the given directory without affecting other files."""
    if directory.exists():
        for wav_file in directory.glob("*.wav"):
            wav_file.unlink()


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def examples_dir(project_root) -> Path:
    return project_root / "examples"


@pytest.fixture(scope="session")
def signals_dir(project_root) -> Path:
    return project_root / "signals"


@pytest.fixture(scope="session", autouse=True)
def ensure_examples(examples_dir) -> Path:
    """Ensure example files exist before running tests."""
    if not examples_dir.exists() or not list(examples_dir.glob("*.wav")):
        print("Generating example files...")
        cleanup_wav_files(examples_dir)
        write_examples(str(examples_dir))
    return examples_dir


@pytest.fixture(scope="session", autouse=True)
def ensure_signals(signals_dir) -> Path:
    """Ensure signal files exist before running tests."""
    if not signals_dir.exists() or not list(signals_dir.glob("*.wav")):
        print("Generating signal files...")
        cleanup_wav_files(signals_dir)
        write_signals(str(signals_dir))
    return signals_dir

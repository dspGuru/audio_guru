import sys
from pathlib import Path

import pytest

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from generate import write_examples


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def examples_dir(project_root):
    return project_root / "examples"


@pytest.fixture(scope="session", autouse=True)
def ensure_examples(examples_dir):
    """Ensure example files exist before running tests."""
    if not examples_dir.exists() or not list(examples_dir.glob("*.wav")):
        print("Generating example files...")
        write_examples(str(examples_dir))
    return examples_dir

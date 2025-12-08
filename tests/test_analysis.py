import pytest
from analysis import Analysis


def test_analysis_import():
    # Verify Analysis is importable and is a class
    assert isinstance(Analysis, type)


def test_cannot_instantiate_abstract():
    # Abstract base class cannot be instantiated
    with pytest.raises(TypeError):
        Analysis()


def test_partial_subclass_missing_to_dict_is_abstract():
    # Subclass missing to_dict remains abstract
    class Partial(Analysis):
        def summary(self) -> str:
            return "summary"

    with pytest.raises(TypeError):
        Partial({})


def test_partial_subclass_missing_summary_is_abstract():
    # Subclass missing summary remains abstract
    class Partial2(Analysis):
        def to_dict(self) -> dict:
            return {}

    with pytest.raises(TypeError):
        Partial2({})


def test_concrete_subclass_implements_and_copies_metadata():
    # Concrete implementation should be instantiable and copy metadata (shallow)
    class DummyAnalysis(Analysis):
        def to_dict(self) -> dict:
            return {"md": self.md}

        def summary(self) -> str:
            return "ok"

    md = {"a": 1}
    ana = DummyAnalysis(md)
    # original metadata mutation of top-level key should not affect stored copy
    md["a"] = 2
    assert ana.md["a"] == 1
    assert isinstance(ana.to_dict(), dict)
    assert isinstance(ana.summary(), str)


def test_metadata_shallow_copy_behavior():
    # The copy used in Analysis.__init__ is a shallow copy
    class DummyAnalysis(Analysis):
        def to_dict(self) -> dict:
            return {"md": self.md}

        def summary(self) -> str:
            return "ok"

    md = {"nested": {"x": 1}}
    ana = DummyAnalysis(md)
    # Mutating nested content should be visible through the shallow copy
    md["nested"]["x"] = 99
    assert ana.md["nested"]["x"] == 99

import pytest
from freq_resp import FreqResp
from types import SimpleNamespace


class DummyComp:
    def __init__(self, pwr: float):
        self.pwr = pwr


class DummyComponents(list):
    def __init__(self, pwrs, md=None, ripple_value=0.0, edges=(300.0, 3000.0)):
        super().__init__([DummyComp(p) for p in pwrs])
        self.md = md or {}
        self._ripple_value = ripple_value
        self._edges = edges

    def ripple(self, lower, upper):
        # mimic real implementation signature; ignore bounds for test
        return self._ripple_value

    def band_edges(self, lower, upper, tol):
        # return preconfigured edges
        return self._edges


def test_freq_resp_import():
    # Verify FreqResp class is importable and is a class/type
    assert isinstance(FreqResp, type)


def test_noise_floor_computation():
    comps = DummyComponents([-55.0, -80.5, -60.0])
    fr = FreqResp(comps)
    assert fr.minimum == pytest.approx(-80.5)


def test_to_dict_and_summary_contents():
    comps = DummyComponents(
        pwrs=[-45.2, -47.8, -50.0],
        md=SimpleNamespace(
            test=True,
            unit_id="Unit123",
            desc="Test FreqResp",
            segment=SimpleNamespace(
                cat=SimpleNamespace(name="TEST"), desc="Segment Desc"
            ),
        ),
        ripple_value=1.234,
        edges=(280.0, 3200.0),
    )
    fr = FreqResp(comps)
    d = fr.to_dict()

    assert d["minimum"] == pytest.approx(min([-45.2, -47.8, -50.0]))
    assert d["ripple"] == pytest.approx(1.234)
    assert d["lower_edge"] == pytest.approx(280.0)
    assert d["upper_edge"] == pytest.approx(3200.0)

    summary = fr.summary()
    # Check formatted values appear in the summary string
    assert "Unit123" in summary
    assert "Test FreqResp" in summary
    assert "TEST" in summary
    assert "280" in summary
    assert "3200" in summary
    assert "1.2" in summary
    assert "-50.0" in summary


def test_empty_components_raises_value_error():
    empty = DummyComponents([], md={})
    with pytest.raises(ValueError):
        FreqResp(empty)

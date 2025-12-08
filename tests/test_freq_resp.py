import pytest
from freq_resp import FreqResp


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
    assert fr.noise_floor == pytest.approx(-80.5)


def test_to_dict_and_summary_contents():
    comps = DummyComponents(
        pwrs=[-45.2, -47.8, -50.0],
        md={"test": True},
        ripple_value=1.234,
        edges=(280.0, 3200.0),
    )
    fr = FreqResp(comps)
    d = fr.to_dict()

    assert d["noise_floor"] == pytest.approx(min([-45.2, -47.8, -50.0]))
    assert d["ripple"] == pytest.approx(1.234)
    assert d["lower_edge"] == pytest.approx(280.0)
    assert d["upper_edge"] == pytest.approx(3200.0)

    summary = fr.summary()
    # Check formatted values appear in the summary string
    assert "Noise Floor=" in summary
    assert "Ripple=" in summary
    assert "Lower Edge=" in summary
    assert "Upper Edge=" in summary
    assert "Noise Floor=-50.0 dB" in summary  # min value formatted with one decimal
    assert "Ripple=1.2 dB" in summary  # ripple formatted to one decimal
    assert "Lower Edge=280.0 Hz" in summary
    assert "Upper Edge=3200.0 Hz" in summary


def test_empty_components_raises_value_error():
    empty = DummyComponents([], md={})
    with pytest.raises(ValueError):
        FreqResp(empty)

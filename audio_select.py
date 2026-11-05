"""Audio Device Selection and Categorization"""

import sounddevice as sd


class AudioSelect:
    """
    Obtain and represent a list of available sound devices.
    Provides sorting by category, name, and API.
    """

    def __init__(self):
        """Initialize AudioSelect by querying available devices and host APIs."""
        self.devices = []
        self._update_devices()

    def _update_devices(self):
        """Fetch devices and enrich with API name and category."""
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        self.devices = []
        for i, dev in enumerate(devices):
            # Create a copy to enrich
            device_info = dict(dev)
            device_info["id"] = i

            # Add API name
            api_idx = device_info["hostapi"]
            device_info["api_name"] = host_apis[api_idx]["name"]

            # Add Category
            device_info["category"] = self._get_category(device_info["name"])

            self.devices.append(device_info)

    def _get_category(self, name: str) -> str:
        """
        Determine device category based on heuristic keyword matching.

        Parameters
        ----------
        name : str
            Device name.

        Returns
        -------
        str
            Category name (Speakers, Microphone, Monitor, Line In, Digital, Other).
        """
        name_lower = name.lower()

        if any(kw in name_lower for kw in ["speaker", "loudspeaker", "headphones"]):
            return "Speakers"
        if any(kw in name_lower for kw in ["mic", "micro"]):
            return "Microphone"
        if any(
            kw in name_lower
            for kw in ["monitor", "display", "nvidia", "intel(r) display"]
        ):
            return "Monitor"
        if any(kw in name_lower for kw in ["digital", "spdif", "iec958"]):
            return "Digital"
        if any(kw in name_lower for kw in ["line in", "input"]):
            return "Line In"

        return "Other"

    def sort(self, by: list[str] | None = None):
        """
        Sort the device list.

        Parameters
        ----------
        by : list[str] | None, optional
            Keys to sort by. Supported: 'category', 'name', 'api_name', 'id'.
            Default is ['category', 'name', 'api_name'].
        """
        if by is None:
            by = ["category", "name", "api_name"]

        # Python's sort is stable, so we sort by the least significant key first
        for key in reversed(by):
            self.devices.sort(key=lambda d: d.get(key, ""))

    def __repr__(self) -> str:
        """Return a formatted string representation of the devices."""
        lines = [f"{'ID':<3} {'Category':<12} {'API':<15} {'Name'}"]
        lines.append("-" * 60)
        for dev in self.devices:
            lines.append(
                f"{dev['id']:<3} {dev['category']:<12} "
                f"{dev['api_name']:<15} {dev['name']}"
            )
        return "\n".join(lines)

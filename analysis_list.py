"""Analysis list module"""

from operator import attrgetter
from pathlib import Path

import pandas as pd


class AnalysisList(list):
    """
    A generic list container for Analysis objects, providing convenience methods
    for sorting, printing summaries, converting to pandas DataFrame, and
    exporting to CSV.
    """

    def filtered(self, **criteria) -> "AnalysisList":
        """
        Return a new AnalysisList filtered by the specified criteria.

        Parameters
        ----------
        criteria : dict
            Key-value pairs to filter the list.

        Returns
        -------
        AnalysisList
            A new AnalysisList containing only the items that match the criteria.

        Examples
        --------
        >>> filtered_list = analysis_list.filtered(fs=44100)
        """
        filtered_list = AnalysisList(
            [
                item
                for item in self
                if all(getattr(item, key) == value for key, value in criteria.items())
            ]
        )
        return filtered_list

    def print(self, sort_stat_attr: str = "") -> None:
        """
        Print the list sorted on the specified statistic attribute.

        Parameters
        ----------
        sort_stat_attr : str, optional
            Name of the statistic attribute to sort by (default is 'thd').
        """
        if self:
            print(self.summary(sort_stat_attr))

    def summary(self, sort_stat_attr: str = "") -> str:
        """
        Returns a summary string for the list.

        Returns
        -------
        str
            Summary string.
        """
        if not self:
            return "Empty AnalysisList"

        summary_lines = [
            type(self[0]).summary_header(),
            "--------------------------------------------------------------------------------",
        ]
        if sort_stat_attr:
            self.sort(key=attrgetter(sort_stat_attr))

        for analysis in self:
            summary_lines.append(analysis.summary())

        return "\n".join(summary_lines)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame representing the list using Stats.get_stats() for
        each stat in the list.
        """
        records = [analysis.to_dict() for analysis in self]
        df = pd.DataFrame.from_records(records)
        return df

    def to_csv(
        self, filepath: str, index: bool = False, print_head: bool = False
    ) -> None:
        """
        Write the object's DataFrame to the specified CSV file.

        Parameters
        ----------
        filepath : str
            Path to the output CSV file.
        index : bool, optional
            Whether to write row indices to the CSV file (default is False).
        """
        df = self.get_dataframe()
        p = Path(filepath)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        # Print the data frame header if specified
        if print_head:
            print(df.head())

        # Write the DataFrame to CSV with specified options
        df.to_csv(p, header=True, index=index, float_format="%.1f")

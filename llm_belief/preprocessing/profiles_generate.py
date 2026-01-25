"""Profile generation for pairwise comparison experiments.

This module provides the ProfileGenerator class for generating
smartphone profile datasets used in LLM belief elicitation experiments.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from llm_belief.utils.attributes import generate_all_profiles
from llm_belief.utils.paths import get_data_path


class ProfileGenerator:
    """Generator for smartphone profile datasets.

    This class provides methods for generating and saving profile datasets
    used in pairwise comparison experiments.

    Attributes:
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: int = 2025):
        """Initialize the generator.

        Args:
            seed: Random seed for profile generation. Defaults to 2025.
        """
        self.seed = seed

    def generate(self) -> List[Dict[str, Any]]:
        """Generate all smartphone profiles.

        Returns:
            List of profile dictionaries with attribute values.
        """
        return generate_all_profiles(seed=self.seed)

    def generate_csv(
        self,
        output_file: Optional[str] = None,
    ) -> Path:
        """Generate profiles and save to CSV.

        Args:
            output_file: Output filename. Defaults to 'profiles_shuffled.csv'.

        Returns:
            Path to the output CSV file.
        """
        if output_file is None:
            output_file = "profiles_shuffled.csv"

        profiles = self.generate()
        df = pd.DataFrame(profiles)

        out_path = get_data_path(output_file)
        df.to_csv(out_path, index=False)

        return out_path

    def get_profile_count(self) -> int:
        """Get the total number of profiles that will be generated.

        Returns:
            Number of profiles based on attribute combinations.
        """
        profiles = self.generate()
        return len(profiles)


def main() -> None:
    generator = ProfileGenerator()
    out_path = generator.generate_csv()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

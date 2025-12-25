from pathlib import Path

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str


class BaseNoodleModel(BaseModel):
    """Base model for all noodle_fts configuration classes.

    Provides YAML serialization/deserialization methods.
    """

    @classmethod
    def from_yaml(cls, yaml_str: str):
        """Load configuration from YAML string.

        Args:
            yaml_str: YAML string representation

        Returns:
            Configuration instance
        """
        return parse_yaml_raw_as(cls, yaml_str)

    @classmethod
    def from_yaml_file(cls, path: str | Path):
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration instance
        """
        path = Path(path)
        return cls.from_yaml(path.read_text())

    def to_yaml(self) -> str:
        """Export configuration to YAML string.

        Returns:
            YAML string representation
        """
        return to_yaml_str(self)

    def to_yaml_file(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to output YAML file
        """
        path = Path(path)
        path.write_text(self.to_yaml())

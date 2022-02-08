from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from yaml import safe_load

ROOT_DIR = Path(__file__).absolute().parent

MAP_GENERATION = "map_generation"


@dataclass
class MapGenerationConfig:

    protein: str
    ligands: List[str]
    identifier: str = "ID"
    label: str = "IC50[uM](SPA)"
    transformation_threshold: int = 3
    atom_mapping_strategy: str = "geometry"
    forcefield: str = str(ROOT_DIR.joinpath("timemachine/ff/params/smirnoff_1_1_0_ccc.py"))
    output: Optional[str] = None
    cores: Optional[List[Dict]] = None
    networks: Optional[List[Dict]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MapGenerationConfig":
        return cls(**data)


@dataclass
class TimemachineConfig:

    map_generation: Optional[MapGenerationConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TimemachineConfig":
        """from_yaml"""
        if not Path(yaml_path).is_file():
            raise FileNotFoundError(yaml_path)
        conf = cls()
        with open(yaml_path, "r") as ifs:
            data = safe_load(ifs)
        if MAP_GENERATION in data:
            conf.map_generation = MapGenerationConfig.from_dict(data[MAP_GENERATION])
        for field in fields(cls):
            if getattr(cls, field.name) is not None:
                break
        return conf

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    url: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def save_sections(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for section in sections:
            output.write(json.dumps(section.to_dict(), ensure_ascii=False) + "\n")


def load_sections(path: Path) -> list[Section]:
    with path.open(encoding="utf-8") as source:
        return [Section(**json.loads(line)) for line in source if line.strip()]

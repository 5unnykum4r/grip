"""Skills system: discover and load SKILL.md files from multiple directories.

Skills are markdown files that provide the agent with specialized knowledge
or workflows. Each SKILL.md contains a name, description, and instructions
that are injected into the system prompt when the skill is loaded.

File format (SKILL.md):
    # Skill Name

    > Description of what this skill does.

    ## Instructions
    Step-by-step instructions for the agent...

Skills are discovered from (in priority order — later sources override earlier):
  1. grip/skills/builtin/   — built-in skills shipped with grip
  2. ~/.agents/skills/      — global shared skills used by multiple agentic tools
  3. workspace/skills/      — user-installed skills specific to this workspace
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(slots=True)
class Skill:
    """A loaded skill with parsed metadata."""

    name: str
    description: str
    content: str
    source_path: Path
    always_loaded: bool = False
    category: str = "general"

    @property
    def display_name(self) -> str:
        return self.name or self.source_path.stem


class SkillsLoader:
    """Discovers and loads SKILL.md files from configured directories."""

    # Global shared skills directory used by multiple agentic tools.
    _GLOBAL_SKILLS_DIR = Path.home() / ".agents" / "skills"

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_skills_dir = workspace_root / "skills"
        self._builtin_skills_dir = Path(__file__).parent / "builtin"
        self._skills: dict[str, Skill] = {}

    def scan(self) -> list[Skill]:
        """Scan all skill directories and load skill metadata.

        Supports two formats:
          - Flat files: skills_dir/*.md
          - Folder-based: skills_dir/skill-name/SKILL.md (preferred)

        Scan order (later overrides earlier):
          1. grip/skills/builtin/   — shipped with grip
          2. ~/.agents/skills/      — global shared skills across agentic tools
          3. workspace/skills/      — workspace-specific user overrides
        """
        self._skills.clear()

        # Built-in skills first (lowest priority)
        if self._builtin_skills_dir.exists():
            self._scan_directory(self._builtin_skills_dir)

        # Global shared skills (override built-ins)
        if self._GLOBAL_SKILLS_DIR.exists():
            self._scan_directory(self._GLOBAL_SKILLS_DIR)

        # Workspace skills (highest priority — override everything)
        if self._workspace_skills_dir.exists():
            self._scan_directory(self._workspace_skills_dir)

        logger.debug("Loaded {} skills", len(self._skills))
        return list(self._skills.values())

    def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for skills in both flat and folder-based formats."""
        # Folder-based: skill-name/SKILL.md
        for path in sorted(directory.glob("*/SKILL.md")):
            skill = self._parse_skill_file(path)
            if skill:
                self._skills[skill.name] = skill

        # Flat files: *.md (directly in the directory, not in subfolders)
        for path in sorted(directory.glob("*.md")):
            if path.is_file():
                skill = self._parse_skill_file(path)
                if skill:
                    self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Skill | None:
        """Get a loaded skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def get_skill_names(self) -> list[str]:
        """Return names of all loaded skills (for ContextBuilder)."""
        return list(self._skills.keys())

    def get_always_loaded_content(self) -> str:
        """Return concatenated content of skills marked always_loaded."""
        parts = []
        for skill in self._skills.values():
            if skill.always_loaded:
                parts.append(f"## Skill: {skill.name}\n\n{skill.content}")
        return "\n\n---\n\n".join(parts)

    def install_skill(self, content: str, filename: str) -> Path:
        """Save a skill file to the workspace skills directory.

        Returns the path of the created file.
        """
        self._workspace_skills_dir.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(".md"):
            filename = f"{filename}.md"
        target = self._workspace_skills_dir / filename
        target.write_text(content, encoding="utf-8")
        logger.info("Skill installed: {}", target)
        return target

    def remove_skill(self, name: str) -> bool:
        """Remove a workspace skill by name. Returns True if removed."""
        skill = self._skills.get(name)
        if not skill:
            return False
        if not str(skill.source_path).startswith(str(self._workspace_skills_dir)):
            logger.warning("Cannot remove built-in skill: {}", name)
            return False
        skill.source_path.unlink(missing_ok=True)
        del self._skills[name]
        return True

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Extract YAML frontmatter from text if present.

        Returns (metadata_dict, remaining_text). If no frontmatter is found,
        returns an empty dict and the original text unchanged.
        Uses simple key: value parsing (no PyYAML dependency).
        """
        stripped = text.lstrip("\n")
        if not stripped.startswith("---"):
            return {}, text

        # Find the closing ---
        end_idx = stripped.find("---", 3)
        if end_idx == -1:
            return {}, text

        yaml_block = stripped[3:end_idx].strip()
        remaining = stripped[end_idx + 3 :].lstrip("\n")

        metadata: dict[str, str] = {}
        for line in yaml_block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            key = line[:colon_pos].strip()
            value = line[colon_pos + 1 :].strip()
            metadata[key] = value
        return metadata, remaining

    @staticmethod
    def _parse_skill_file(path: Path) -> Skill | None:
        """Parse a SKILL.md file into a Skill object.

        Supports two formats:
          1. YAML frontmatter (preferred):
             ---
             title: Skill Name
             description: What it does
             category: automation
             always_loaded: true
             ---
             (content below)

          2. Legacy H1 + blockquote (backward compatible):
             # Skill Name
             > Description.
             <!-- always_loaded -->
             ... rest is content
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read skill file {}: {}", path, exc)
            return None

        lines = text.strip().splitlines()
        if not lines:
            return None

        # Default name from directory or filename
        name = path.stem
        if path.name == "SKILL.md":
            name = path.parent.name

        # Try YAML frontmatter first
        frontmatter, remaining = SkillsLoader._parse_frontmatter(text)
        if frontmatter:
            fm_name = frontmatter.get("title") or frontmatter.get("name") or name
            fm_desc = frontmatter.get("description", "")
            fm_category = frontmatter.get("category", "general")
            fm_always = frontmatter.get("always_loaded", "false").lower() in ("true", "yes", "1")
            return Skill(
                name=fm_name,
                description=fm_desc,
                content=remaining.strip(),
                source_path=path,
                always_loaded=fm_always,
                category=fm_category,
            )

        # Fallback: legacy H1 + blockquote parsing
        description = ""
        always_loaded = False
        category = "general"
        content_start = 0
        found_name = False
        in_code_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            if not found_name and stripped.startswith("# ") and not stripped.startswith("## "):
                name = stripped[2:].strip()
                content_start = i + 1
                found_name = True
            elif stripped.startswith(">") and not description:
                description = stripped.lstrip("> ").strip()
            elif "always_loaded" in stripped.lower():
                always_loaded = True

            if i > 15:
                break

        content = "\n".join(lines[content_start:]).strip()

        return Skill(
            name=name,
            description=description,
            content=content,
            source_path=path,
            always_loaded=always_loaded,
            category=category,
        )

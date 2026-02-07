# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Skill loader for loading skills from the filesystem or from in-memory content.

Skills can be loaded from directories with the following structure:

    my-skill/
        SKILL.md          # Required: frontmatter (YAML) + instructions (Markdown)
        references/       # Optional: additional markdown instruction files
        assets/           # Optional: resource files (schemas, templates, etc.)
        scripts/          # Optional: executable script files

Or from in-memory content using ``load_skill_from_content()`` for environments
without filesystem access (e.g., serverless, remote agents).

The SKILL.md file uses YAML frontmatter delimited by '---' lines, followed by
the instruction body in Markdown format:

    ---
    name: my-skill
    description: What this skill does and when to use it
    license: Apache-2.0
    ---

    # Instructions

    Follow these steps to ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import Frontmatter, Resources, Script, Skill

logger = logging.getLogger("google_adk." + __name__)

SKILL_FILENAME = "SKILL.md"


def _parse_skill_md(content: str) -> tuple[dict, str]:
  """Parses a SKILL.md file into frontmatter dict and instruction body.

  Args:
    content: The raw text content of a SKILL.md file.

  Returns:
    A tuple of (frontmatter_dict, instructions_body).

  Raises:
    ValueError: If the frontmatter delimiters are missing or malformed.
  """
  content = content.strip()

  if not content.startswith("---"):
    raise ValueError(
        "SKILL.md must start with '---' frontmatter delimiter."
    )

  # Find the closing '---' delimiter (skip the opening one)
  end_index = content.index("---", 3)
  if end_index == -1:
    raise ValueError(
        "SKILL.md is missing the closing '---' frontmatter delimiter."
    )

  frontmatter_text = content[3:end_index].strip()
  instructions = content[end_index + 3:].strip()

  frontmatter_dict = yaml.safe_load(frontmatter_text)
  if not isinstance(frontmatter_dict, dict):
    raise ValueError("SKILL.md frontmatter must be a YAML mapping.")

  return frontmatter_dict, instructions


def _load_resources(skill_dir: Path) -> Resources:
  """Loads L3 resources from a skill directory.

  Scans for references/, assets/, and scripts/ subdirectories and loads
  their contents.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    A Resources object with loaded content.
  """
  references = {}
  assets = {}
  scripts = {}

  references_dir = skill_dir / "references"
  if references_dir.is_dir():
    for file_path in sorted(references_dir.rglob("*")):
      if file_path.is_file():
        try:
          key = str(file_path.relative_to(references_dir))
          references[key] = file_path.read_text(encoding="utf-8")
        except Exception as e:
          logger.warning(
              "Failed to read reference file %s: %s", file_path, e
          )

  assets_dir = skill_dir / "assets"
  if assets_dir.is_dir():
    for file_path in sorted(assets_dir.rglob("*")):
      if file_path.is_file():
        try:
          key = str(file_path.relative_to(assets_dir))
          assets[key] = file_path.read_text(encoding="utf-8")
        except Exception as e:
          logger.warning("Failed to read asset file %s: %s", file_path, e)

  scripts_dir = skill_dir / "scripts"
  if scripts_dir.is_dir():
    for file_path in sorted(scripts_dir.rglob("*")):
      if file_path.is_file():
        try:
          key = str(file_path.relative_to(scripts_dir))
          scripts[key] = Script(
              src=file_path.read_text(encoding="utf-8")
          )
        except Exception as e:
          logger.warning(
              "Failed to read script file %s: %s", file_path, e
          )

  return Resources(
      references=references,
      assets=assets,
      scripts=scripts,
  )


def load_skill(skill_dir: Path) -> Skill:
  """Loads a single skill from a directory.

  The directory must contain a SKILL.md file with YAML frontmatter and
  Markdown instructions. Optionally, it may contain references/, assets/,
  and scripts/ subdirectories with additional resources.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    A Skill object.

  Raises:
    FileNotFoundError: If the SKILL.md file is not found.
    ValueError: If the SKILL.md file is malformed.
  """
  skill_dir = Path(skill_dir)
  skill_file = skill_dir / SKILL_FILENAME

  if not skill_file.exists():
    raise FileNotFoundError(
        f"Skill file not found: {skill_file}. "
        f"Each skill directory must contain a {SKILL_FILENAME} file."
    )

  content = skill_file.read_text(encoding="utf-8")
  frontmatter_dict, instructions = _parse_skill_md(content)

  frontmatter = Frontmatter(**frontmatter_dict)
  resources = _load_resources(skill_dir)

  return Skill(
      frontmatter=frontmatter,
      instructions=instructions,
      resources=resources,
  )


def load_skill_from_content(
    skill_md_content: str,
    *,
    resources: Optional[Resources] = None,
) -> Skill:
  """Creates a Skill from SKILL.md content string without filesystem access.

  This is useful for environments where the agent doesn't have filesystem
  access, such as serverless deployments or remote agents. The skill content
  can be fetched from a database, API, or embedded directly in code.

  Args:
    skill_md_content: The raw text content of a SKILL.md file (frontmatter +
      instructions).
    resources: Optional pre-built Resources object. Since there's no
      filesystem, resources must be constructed in memory if needed.

  Returns:
    A Skill object.

  Raises:
    ValueError: If the skill_md_content is malformed.
  """
  frontmatter_dict, instructions = _parse_skill_md(skill_md_content)
  frontmatter = Frontmatter(**frontmatter_dict)

  return Skill(
      frontmatter=frontmatter,
      instructions=instructions,
      resources=resources or Resources(),
  )


def load_skill_from_dict(
    skill_dict: Dict[str, Any],
) -> Skill:
  """Creates a Skill from a dictionary without filesystem access.

  This is useful for loading skills from APIs, databases, or configuration
  systems in environments without filesystem access.

  Args:
    skill_dict: A dictionary with the following keys:
      - name (str): Skill name in kebab-case (required).
      - description (str): What the skill does (required).
      - instructions (str): The skill instructions in Markdown (required).
      - license (str, optional): License for the skill.
      - compatibility (str, optional): Compatibility information.
      - allowed_tools (str, optional): Tool patterns the skill requires.
      - metadata (dict, optional): Key-value pairs for client properties.
      - resources (dict, optional): Dict with 'references', 'assets',
        'scripts' keys, each mapping IDs to content strings.

  Returns:
    A Skill object.
  """
  frontmatter_kwargs = {
      "name": skill_dict["name"],
      "description": skill_dict["description"],
  }
  for optional_key in ("license", "compatibility", "allowed_tools", "metadata"):
    if optional_key in skill_dict:
      frontmatter_kwargs[optional_key] = skill_dict[optional_key]

  frontmatter = Frontmatter(**frontmatter_kwargs)

  resources = Resources()
  if "resources" in skill_dict:
    res = skill_dict["resources"]
    resources = Resources(
        references=res.get("references", {}),
        assets=res.get("assets", {}),
        scripts={
            k: Script(src=v) for k, v in res.get("scripts", {}).items()
        },
    )

  return Skill(
      frontmatter=frontmatter,
      instructions=skill_dict["instructions"],
      resources=resources,
  )


def load_skills(
    skills_dir: Path,
    *,
    skill_names: Optional[List[str]] = None,
) -> list[Skill]:
  """Loads all skills from a parent directory.

  Each subdirectory containing a SKILL.md file is treated as a skill.

  Args:
    skills_dir: Path to the parent directory containing skill subdirectories.
    skill_names: Optional list of skill names to load. If provided, only
      skills whose directory names match will be loaded.

  Returns:
    A list of loaded Skill objects.
  """
  skills_dir = Path(skills_dir)
  skills = []

  if not skills_dir.is_dir():
    logger.warning("Skills directory does not exist: %s", skills_dir)
    return skills

  for entry in sorted(skills_dir.iterdir()):
    if not entry.is_dir():
      continue

    if skill_names and entry.name not in skill_names:
      continue

    skill_file = entry / SKILL_FILENAME
    if not skill_file.exists():
      continue

    try:
      skill = load_skill(entry)
      skills.append(skill)
    except Exception as e:
      logger.warning("Failed to load skill from %s: %s", entry, e)

  return skills

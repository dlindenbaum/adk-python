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

"""Unit tests for skill loader."""

from pathlib import Path

from google.adk.skills.loader import _parse_skill_md
from google.adk.skills.loader import load_skill
from google.adk.skills.loader import load_skill_from_content
from google.adk.skills.loader import load_skill_from_dict
from google.adk.skills.loader import load_skills
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
import pytest


class TestParseSkillMd:

  def test_basic_frontmatter_and_body(self):
    content = """---
name: test-skill
description: A test skill
---

# Instructions

Do the thing."""
    frontmatter, instructions = _parse_skill_md(content)
    assert frontmatter["name"] == "test-skill"
    assert frontmatter["description"] == "A test skill"
    assert "# Instructions" in instructions
    assert "Do the thing." in instructions

  def test_frontmatter_with_optional_fields(self):
    content = """---
name: advanced-skill
description: An advanced skill
license: Apache-2.0
compatibility: gemini-2.0
allowed_tools: "bash,python"
metadata:
  author: test
  version: "1.0"
---

Follow these steps."""
    frontmatter, instructions = _parse_skill_md(content)
    assert frontmatter["name"] == "advanced-skill"
    assert frontmatter["license"] == "Apache-2.0"
    assert frontmatter["metadata"]["author"] == "test"
    assert "Follow these steps." in instructions

  def test_missing_opening_delimiter(self):
    content = "name: test\n---\nInstructions"
    with pytest.raises(ValueError, match="must start with '---'"):
      _parse_skill_md(content)

  def test_missing_closing_delimiter(self):
    content = "---\nname: test\nInstructions without closing"
    with pytest.raises(ValueError):
      _parse_skill_md(content)

  def test_empty_instructions(self):
    content = """---
name: minimal-skill
description: Minimal
---"""
    frontmatter, instructions = _parse_skill_md(content)
    assert frontmatter["name"] == "minimal-skill"
    assert instructions == ""

  def test_non_dict_frontmatter(self):
    content = """---
- item1
- item2
---

Instructions"""
    with pytest.raises(ValueError, match="must be a YAML mapping"):
      _parse_skill_md(content)


class TestLoadSkill:

  def test_load_basic_skill(self, tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: my-skill
description: My test skill
---

# Instructions

Step 1: Do something.
Step 2: Do something else.""")

    skill = load_skill(skill_dir)
    assert skill.name == "my-skill"
    assert skill.description == "My test skill"
    assert "Step 1:" in skill.instructions
    assert "Step 2:" in skill.instructions

  def test_load_skill_with_resources(self, tmp_path):
    skill_dir = tmp_path / "resource-skill"
    skill_dir.mkdir()

    # Create SKILL.md
    (skill_dir / "SKILL.md").write_text("""---
name: resource-skill
description: Skill with resources
---

Use the resources.""")

    # Create references
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()
    (refs_dir / "guide.md").write_text("# Guide\nFollow this guide.")

    # Create assets
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "schema.json").write_text('{"type": "object"}')

    # Create scripts
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "setup.sh").write_text("#!/bin/bash\necho 'hello'")

    skill = load_skill(skill_dir)
    assert skill.name == "resource-skill"
    assert skill.resources.get_reference("guide.md") == "# Guide\nFollow this guide."
    assert skill.resources.get_asset("schema.json") == '{"type": "object"}'
    assert str(skill.resources.get_script("setup.sh")) == "#!/bin/bash\necho 'hello'"

  def test_load_skill_with_nested_resources(self, tmp_path):
    skill_dir = tmp_path / "nested-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: nested-skill
description: Skill with nested resources
---

Instructions.""")

    # Create nested references
    refs_dir = skill_dir / "references" / "subdir"
    refs_dir.mkdir(parents=True)
    (refs_dir / "nested.md").write_text("Nested content")

    skill = load_skill(skill_dir)
    assert skill.resources.get_reference("subdir/nested.md") == "Nested content"

  def test_load_skill_missing_skill_md(self, tmp_path):
    skill_dir = tmp_path / "empty-skill"
    skill_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Skill file not found"):
      load_skill(skill_dir)

  def test_load_skill_no_resources_dirs(self, tmp_path):
    skill_dir = tmp_path / "minimal-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: minimal
description: Minimal skill
---

Just instructions.""")

    skill = load_skill(skill_dir)
    assert skill.resources.references == {}
    assert skill.resources.assets == {}
    assert skill.resources.scripts == {}


class TestLoadSkills:

  def test_load_multiple_skills(self, tmp_path):
    # Create two skill directories
    for name in ["skill-a", "skill-b"]:
      skill_dir = tmp_path / name
      skill_dir.mkdir()
      (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Description of {name}
---

Instructions for {name}.""")

    skills = load_skills(tmp_path)
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"skill-a", "skill-b"}

  def test_load_skills_with_filter(self, tmp_path):
    for name in ["skill-a", "skill-b", "skill-c"]:
      skill_dir = tmp_path / name
      skill_dir.mkdir()
      (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Description of {name}
---

Instructions for {name}.""")

    skills = load_skills(tmp_path, skill_names=["skill-a", "skill-c"])
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"skill-a", "skill-c"}

  def test_load_skills_skips_non_directories(self, tmp_path):
    # Create a skill directory
    skill_dir = tmp_path / "real-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: real-skill
description: A real skill
---

Instructions.""")

    # Create a non-directory file
    (tmp_path / "not-a-skill.txt").write_text("I'm not a skill")

    skills = load_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "real-skill"

  def test_load_skills_skips_dirs_without_skill_md(self, tmp_path):
    # Create a directory without SKILL.md
    (tmp_path / "empty-dir").mkdir()

    # Create a proper skill directory
    skill_dir = tmp_path / "proper-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("""---
name: proper-skill
description: Proper skill
---

Instructions.""")

    skills = load_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "proper-skill"

  def test_load_skills_nonexistent_directory(self, tmp_path):
    skills = load_skills(tmp_path / "nonexistent")
    assert skills == []

  def test_load_skills_empty_directory(self, tmp_path):
    skills = load_skills(tmp_path)
    assert skills == []

  def test_load_skills_continues_on_error(self, tmp_path):
    # Create a valid skill
    valid_dir = tmp_path / "valid-skill"
    valid_dir.mkdir()
    (valid_dir / "SKILL.md").write_text("""---
name: valid-skill
description: Valid
---

Instructions.""")

    # Create an invalid skill (bad frontmatter)
    invalid_dir = tmp_path / "invalid-skill"
    invalid_dir.mkdir()
    (invalid_dir / "SKILL.md").write_text("Not valid frontmatter")

    skills = load_skills(tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "valid-skill"


class TestLoadSkillFromContent:

  def test_basic_content(self):
    content = """---
name: api-skill
description: Skill loaded from content
---

# API Skill Instructions

Call the API with the correct parameters."""
    skill = load_skill_from_content(content)
    assert skill.name == "api-skill"
    assert skill.description == "Skill loaded from content"
    assert "Call the API" in skill.instructions
    assert skill.resources.references == {}

  def test_with_resources(self):
    content = """---
name: resource-skill
description: Skill with in-memory resources
---

Use the provided resources."""
    resources = Resources(
        references={"guide.md": "# Guide"},
        assets={"schema.json": '{"type": "object"}'},
        scripts={"run.sh": Script(src="echo hello")},
    )
    skill = load_skill_from_content(content, resources=resources)
    assert skill.name == "resource-skill"
    assert skill.resources.get_reference("guide.md") == "# Guide"
    assert skill.resources.get_asset("schema.json") == '{"type": "object"}'
    assert str(skill.resources.get_script("run.sh")) == "echo hello"

  def test_invalid_content(self):
    with pytest.raises(ValueError):
      load_skill_from_content("not valid content")


class TestLoadSkillFromDict:

  def test_minimal_dict(self):
    skill = load_skill_from_dict({
        "name": "dict-skill",
        "description": "A skill from a dict",
        "instructions": "Do the thing.",
    })
    assert skill.name == "dict-skill"
    assert skill.description == "A skill from a dict"
    assert skill.instructions == "Do the thing."
    assert skill.resources.references == {}

  def test_full_dict(self):
    skill = load_skill_from_dict({
        "name": "full-skill",
        "description": "Full skill from dict",
        "instructions": "Follow all steps.",
        "license": "MIT",
        "compatibility": "gemini-2.0",
        "allowed_tools": "bash",
        "metadata": {"author": "test", "version": "1.0"},
        "resources": {
            "references": {"guide.md": "# Guide"},
            "assets": {"config.yaml": "key: value"},
            "scripts": {"setup.sh": "echo hello"},
        },
    })
    assert skill.name == "full-skill"
    assert skill.frontmatter.license == "MIT"
    assert skill.frontmatter.metadata == {"author": "test", "version": "1.0"}
    assert skill.resources.get_reference("guide.md") == "# Guide"
    assert skill.resources.get_asset("config.yaml") == "key: value"
    assert str(skill.resources.get_script("setup.sh")) == "echo hello"

  def test_dict_without_resources(self):
    skill = load_skill_from_dict({
        "name": "no-resources",
        "description": "No resources skill",
        "instructions": "Simple instructions.",
    })
    assert skill.resources.references == {}
    assert skill.resources.assets == {}
    assert skill.resources.scripts == {}

  def test_dict_missing_required_fields(self):
    with pytest.raises(KeyError):
      load_skill_from_dict({"name": "incomplete"})

  def test_dict_with_partial_resources(self):
    skill = load_skill_from_dict({
        "name": "partial",
        "description": "Partial resources",
        "instructions": "Instructions.",
        "resources": {
            "references": {"ref.md": "Reference content"},
        },
    })
    assert skill.resources.get_reference("ref.md") == "Reference content"
    assert skill.resources.assets == {}
    assert skill.resources.scripts == {}

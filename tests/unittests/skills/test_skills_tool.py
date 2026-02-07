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

"""Unit tests for SkillsTool."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.skills.skills_tool import _skill_name_to_tool_name
from google.adk.skills.skills_tool import SkillsTool
from google.adk.tools.tool_context import ToolContext
import pytest


def _make_skill(
    name="test-skill",
    description="A test skill",
    instructions="Do the thing.",
    resources=None,
):
  return Skill(
      frontmatter=Frontmatter(name=name, description=description),
      instructions=instructions,
      resources=resources or Resources(),
  )


async def _make_tool_context():
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  agent = SequentialAgent(name="test_agent")
  invocation_context = InvocationContext(
      invocation_id="test_id",
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ToolContext(invocation_context)


class TestSkillNameConversion:

  def test_kebab_to_snake(self):
    assert _skill_name_to_tool_name("my-skill") == "my_skill"

  def test_no_hyphens(self):
    assert _skill_name_to_tool_name("myskill") == "myskill"

  def test_multiple_hyphens(self):
    assert _skill_name_to_tool_name("my-cool-skill") == "my_cool_skill"


class TestSkillsTool:

  def test_initialization(self):
    skill = _make_skill()
    tool = SkillsTool(skill)
    assert tool.name == "test_skill"
    assert tool.description == "A test skill"
    assert tool.skill is skill

  def test_name_conversion(self):
    skill = _make_skill(name="my-complex-skill")
    tool = SkillsTool(skill)
    assert tool.name == "my_complex_skill"

  def test_get_declaration_no_resources(self):
    skill = _make_skill()
    tool = SkillsTool(skill)
    declaration = tool._get_declaration()
    assert declaration is not None
    assert declaration.name == "test_skill"
    assert declaration.description == "A test skill"
    # No resource_id parameter when no resources
    if declaration.parameters and declaration.parameters.properties:
      assert "resource_id" not in declaration.parameters.properties

  def test_get_declaration_with_resources(self):
    resources = Resources(
        references={"guide.md": "Guide content"},
        assets={"schema.json": '{"type": "object"}'},
        scripts={"setup.sh": Script(src="echo hello")},
    )
    skill = _make_skill(resources=resources)
    tool = SkillsTool(skill)
    declaration = tool._get_declaration()
    assert declaration is not None
    assert "resource_id" in declaration.parameters.properties

  @pytest.mark.asyncio
  async def test_run_async_basic(self):
    skill = _make_skill(instructions="Step 1: Do this.\nStep 2: Do that.")
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    result = await tool.run_async(args={}, tool_context=tool_context)

    assert result["skill_name"] == "test-skill"
    assert "Step 1: Do this." in result["instructions"]
    assert "Step 2: Do that." in result["instructions"]

  @pytest.mark.asyncio
  async def test_run_async_with_resource_request(self):
    resources = Resources(
        references={"guide.md": "Guide content"},
        assets={"schema.json": '{"type": "object"}'},
    )
    skill = _make_skill(resources=resources)
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    # Request a specific reference
    result = await tool.run_async(
        args={"resource_id": "guide.md"}, tool_context=tool_context
    )
    assert result["resource"] == "Guide content"
    assert "available_resources" in result

    # Request a specific asset
    result = await tool.run_async(
        args={"resource_id": "schema.json"}, tool_context=tool_context
    )
    assert result["resource"] == '{"type": "object"}'

  @pytest.mark.asyncio
  async def test_run_async_with_script_request(self):
    resources = Resources(
        scripts={"setup.sh": Script(src="#!/bin/bash\necho hello")},
    )
    skill = _make_skill(resources=resources)
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    result = await tool.run_async(
        args={"resource_id": "setup.sh"}, tool_context=tool_context
    )
    assert "#!/bin/bash" in result["resource"]

  @pytest.mark.asyncio
  async def test_run_async_with_nonexistent_resource(self):
    resources = Resources(
        references={"guide.md": "Guide content"},
    )
    skill = _make_skill(resources=resources)
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    result = await tool.run_async(
        args={"resource_id": "nonexistent.md"}, tool_context=tool_context
    )
    assert "resource_error" in result
    assert "nonexistent.md" in result["resource_error"]

  @pytest.mark.asyncio
  async def test_run_async_includes_available_resources(self):
    resources = Resources(
        references={"ref1.md": "content1"},
        assets={"asset1.json": "content2"},
        scripts={"script1.sh": Script(src="echo hi")},
    )
    skill = _make_skill(resources=resources)
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    result = await tool.run_async(args={}, tool_context=tool_context)
    assert "available_resources" in result
    assert "ref1.md" in result["available_resources"]["references"]
    assert "asset1.json" in result["available_resources"]["assets"]
    assert "script1.sh" in result["available_resources"]["scripts"]

  @pytest.mark.asyncio
  async def test_run_async_no_available_resources_key_when_empty(self):
    skill = _make_skill()
    tool = SkillsTool(skill)
    tool_context = await _make_tool_context()

    result = await tool.run_async(args={}, tool_context=tool_context)
    assert "available_resources" not in result

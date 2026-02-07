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

"""Unit tests for SkillsToolset."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.skills.skills_tool import SkillsTool
from google.adk.skills.skills_toolset import SkillsToolset
from google.adk.tools.tool_context import ToolContext
import pytest


def _make_skill(name="test-skill", description="A test skill"):
  return Skill(
      frontmatter=Frontmatter(name=name, description=description),
      instructions=f"Instructions for {name}.",
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


class TestSkillsToolset:

  @pytest.mark.asyncio
  async def test_empty_toolset(self):
    toolset = SkillsToolset(skills=[])
    tools = await toolset.get_tools()
    assert tools == []

  @pytest.mark.asyncio
  async def test_single_skill(self):
    skill = _make_skill()
    toolset = SkillsToolset(skills=[skill])
    tools = await toolset.get_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], SkillsTool)
    assert tools[0].name == "test_skill"

  @pytest.mark.asyncio
  async def test_multiple_skills(self):
    skills = [
        _make_skill(name="skill-a", description="Skill A"),
        _make_skill(name="skill-b", description="Skill B"),
        _make_skill(name="skill-c", description="Skill C"),
    ]
    toolset = SkillsToolset(skills=skills)
    tools = await toolset.get_tools()
    assert len(tools) == 3
    names = {tool.name for tool in tools}
    assert names == {"skill_a", "skill_b", "skill_c"}

  @pytest.mark.asyncio
  async def test_tool_filter_with_list(self):
    skills = [
        _make_skill(name="skill-a", description="Skill A"),
        _make_skill(name="skill-b", description="Skill B"),
    ]
    toolset = SkillsToolset(skills=skills, tool_filter=["skill_a"])

    tool_context = await _make_tool_context()
    # Use the readonly_context from tool_context's invocation_context
    from google.adk.agents.readonly_context import ReadonlyContext

    readonly_ctx = ReadonlyContext(tool_context._invocation_context)
    tools = await toolset.get_tools(readonly_context=readonly_ctx)
    assert len(tools) == 1
    assert tools[0].name == "skill_a"

  @pytest.mark.asyncio
  async def test_get_tools_returns_copies(self):
    skill = _make_skill()
    toolset = SkillsToolset(skills=[skill])
    tools1 = await toolset.get_tools()
    tools2 = await toolset.get_tools()
    # Should return separate list objects
    assert tools1 is not tools2

  @pytest.mark.asyncio
  async def test_process_llm_request_injects_skills_info(self):
    skills = [
        _make_skill(name="skill-a", description="Does A things"),
        _make_skill(name="skill-b", description="Does B things"),
    ]
    toolset = SkillsToolset(skills=skills)
    tool_context = await _make_tool_context()
    llm_request = LlmRequest()

    await toolset.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    # Verify system instruction was injected
    assert llm_request.config is not None
    system_instruction = llm_request.config.system_instruction
    assert system_instruction is not None
    assert "available_skills" in system_instruction
    assert "skill-a" in system_instruction
    assert "Does A things" in system_instruction
    assert "skill-b" in system_instruction
    assert "Does B things" in system_instruction

  @pytest.mark.asyncio
  async def test_process_llm_request_appends_to_existing_instruction(self):
    skills = [_make_skill(name="my-skill", description="My skill")]
    toolset = SkillsToolset(skills=skills)
    tool_context = await _make_tool_context()

    from google.genai import types

    llm_request = LlmRequest()
    llm_request.config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant."
    )

    await toolset.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    system_instruction = llm_request.config.system_instruction
    assert "You are a helpful assistant." in system_instruction
    assert "available_skills" in system_instruction

  @pytest.mark.asyncio
  async def test_process_llm_request_empty_skills_noop(self):
    toolset = SkillsToolset(skills=[])
    tool_context = await _make_tool_context()
    llm_request = LlmRequest()

    await toolset.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    # Should not modify the system instruction
    assert (
        llm_request.config is None
        or llm_request.config.system_instruction is None
    )

  @pytest.mark.asyncio
  async def test_from_directory(self, tmp_path):
    # Create skill directories
    for name in ["skill-x", "skill-y"]:
      skill_dir = tmp_path / name
      skill_dir.mkdir()
      (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Description of {name}
---

Instructions for {name}.""")

    toolset = SkillsToolset.from_directory(tmp_path)
    assert len(toolset.skills) == 2
    tools = await toolset.get_tools()
    assert len(tools) == 2

  @pytest.mark.asyncio
  async def test_from_directory_with_filter(self, tmp_path):
    for name in ["skill-x", "skill-y", "skill-z"]:
      skill_dir = tmp_path / name
      skill_dir.mkdir()
      (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Description of {name}
---

Instructions for {name}.""")

    toolset = SkillsToolset.from_directory(
        tmp_path, skill_names=["skill-x", "skill-z"]
    )
    assert len(toolset.skills) == 2
    names = {s.name for s in toolset.skills}
    assert names == {"skill-x", "skill-z"}

  @pytest.mark.asyncio
  async def test_from_directory_empty(self, tmp_path):
    toolset = SkillsToolset.from_directory(tmp_path)
    assert len(toolset.skills) == 0
    tools = await toolset.get_tools()
    assert tools == []

  @pytest.mark.asyncio
  async def test_from_directory_nonexistent(self, tmp_path):
    toolset = SkillsToolset.from_directory(tmp_path / "nonexistent")
    assert len(toolset.skills) == 0

  @pytest.mark.asyncio
  async def test_close(self):
    toolset = SkillsToolset(skills=[_make_skill()])
    # Should not raise
    await toolset.close()

  @pytest.mark.asyncio
  async def test_tool_name_prefix(self):
    skills = [_make_skill(name="my-skill")]
    toolset = SkillsToolset(skills=skills, tool_name_prefix="prefix")
    tools = await toolset.get_tools_with_prefix()
    assert len(tools) == 1
    assert tools[0].name == "prefix_my_skill"

  @pytest.mark.asyncio
  async def test_skills_attribute(self):
    skills = [
        _make_skill(name="skill-a"),
        _make_skill(name="skill-b"),
    ]
    toolset = SkillsToolset(skills=skills)
    assert len(toolset.skills) == 2
    assert toolset.skills[0].name == "skill-a"
    assert toolset.skills[1].name == "skill-b"

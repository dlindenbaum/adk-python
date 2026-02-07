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

"""Toolset that exposes skills as tools for LLM agents.

The SkillsToolset loads skills from the filesystem or from pre-loaded Skill
objects, and exposes each skill as a callable tool. It also injects skill
discovery information into the LLM system instruction so the model knows
which skills are available.

Usage:

    from google.adk.skills import SkillsToolset

    # From a directory of skills
    toolset = SkillsToolset.from_directory("/path/to/skills")

    # From pre-loaded Skill objects
    toolset = SkillsToolset(skills=[skill1, skill2])

    # Use with an agent
    agent = Agent(
        name="my_agent",
        tools=[toolset],
        ...
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Union

from ..agents.readonly_context import ReadonlyContext
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from .loader import load_skills
from .models import Skill
from .prompt import format_skills_as_xml
from .skills_tool import SkillsTool

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class SkillsToolset(BaseToolset):
  """A toolset that exposes skills as callable tools for LLM agents.

  Each skill is converted into a SkillsTool that the LLM can invoke to
  retrieve skill instructions and resources. The toolset also injects
  a skill discovery prompt into the LLM system instruction so the model
  knows which skills are available and when to use them.

  Attributes:
    skills: The list of loaded Skill objects.
  """

  def __init__(
      self,
      *,
      skills: List[Skill],
      tool_filter: Optional[Union["ToolPredicate", List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ):
    """Initializes the SkillsToolset.

    Args:
      skills: A list of Skill objects to expose as tools.
      tool_filter: Optional filter for selecting which skill tools to expose.
      tool_name_prefix: Optional prefix for tool names.
    """
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self.skills = skills
    self._tools: list[SkillsTool] = [SkillsTool(skill) for skill in skills]

  @classmethod
  def from_directory(
      cls,
      skills_dir: Union[str, Path],
      *,
      skill_names: Optional[List[str]] = None,
      tool_filter: Optional[Union["ToolPredicate", List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ) -> SkillsToolset:
    """Creates a SkillsToolset by loading skills from a directory.

    Args:
      skills_dir: Path to the directory containing skill subdirectories.
      skill_names: Optional list of skill directory names to load.
      tool_filter: Optional filter for selecting which skill tools to expose.
      tool_name_prefix: Optional prefix for tool names.

    Returns:
      A SkillsToolset with the loaded skills.
    """
    skills = load_skills(
        Path(skills_dir),
        skill_names=skill_names,
    )
    return cls(
        skills=skills,
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )

  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Returns the list of skill tools, applying any filters.

    Args:
      readonly_context: Optional context for filtering tools.

    Returns:
      A list of SkillsTool instances.
    """
    if not self.tool_filter:
      return list(self._tools)

    if readonly_context is None:
      return list(self._tools)

    return [
        tool
        for tool in self._tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Injects skill discovery information into the LLM system instruction.

    This adds an XML block listing available skills and their descriptions
    to the system instruction, so the model knows which skills can be
    invoked.

    Args:
      tool_context: The tool context.
      llm_request: The outgoing LLM request.
    """
    if not self.skills:
      return

    frontmatters = [skill.frontmatter for skill in self.skills]
    skills_xml = format_skills_as_xml(frontmatters)

    skill_system_prompt = (
        "\n\nYou have access to the following skills. "
        "When a user request matches a skill's description, use the "
        "corresponding tool to retrieve detailed instructions for that skill, "
        "then follow those instructions to complete the task.\n\n"
        + skills_xml
    )

    if llm_request.config is None:
      from google.genai import types

      llm_request.config = types.GenerateContentConfig()

    if llm_request.config.system_instruction is None:
      llm_request.config.system_instruction = skill_system_prompt
    elif isinstance(llm_request.config.system_instruction, str):
      llm_request.config.system_instruction += skill_system_prompt
    else:
      # system_instruction is types.ContentUnion - append as text part
      from google.genai import types

      if isinstance(llm_request.config.system_instruction, types.Content):
        llm_request.config.system_instruction.parts.append(
            types.Part.from_text(text=skill_system_prompt)
        )

  async def close(self) -> None:
    """Performs cleanup."""
    pass

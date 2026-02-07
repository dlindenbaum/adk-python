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

"""Tool that wraps a Skill for use by an LLM agent.

A SkillsTool exposes a loaded Skill as a callable tool. When the LLM decides
to use a skill, it calls this tool to retrieve the skill's detailed
instructions and optionally specific resources.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Optional

from google.genai import types
from typing_extensions import override

from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .models import Skill

logger = logging.getLogger("google_adk." + __name__)


class SkillsTool(BaseTool):
  """A tool that wraps a Skill and returns its instructions when invoked.

  When the LLM calls this tool, it receives the skill's L2 instructions
  and can optionally request specific L3 resources (references, assets,
  or scripts).

  Attributes:
    skill: The Skill object this tool wraps.
  """

  def __init__(self, skill: Skill):
    """Initializes the SkillsTool.

    Args:
      skill: The Skill object to wrap as a tool.
    """
    super().__init__(
        name=_skill_name_to_tool_name(skill.name),
        description=skill.description,
    )
    self.skill = skill

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    properties = {}
    required = []

    # Add resource_id parameter if the skill has L3 resources
    has_resources = (
        self.skill.resources.references
        or self.skill.resources.assets
        or self.skill.resources.scripts
    )
    if has_resources:
      resource_ids = (
          self.skill.resources.list_references()
          + self.skill.resources.list_assets()
          + self.skill.resources.list_scripts()
      )
      properties["resource_id"] = types.Schema(
          type=types.Type.STRING,
          description=(
              "Optional ID of a specific resource to retrieve. "
              f"Available resources: {', '.join(resource_ids)}"
          ),
      )

    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties=properties if properties else None,
            required=required if required else None,
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Returns the skill's instructions and optionally a specific resource.

    Args:
      args: Optional arguments including 'resource_id' to fetch a specific
        resource.
      tool_context: The tool context.

    Returns:
      A dict containing the skill instructions and optionally the requested
      resource content.
    """
    result: dict[str, Any] = {
        "skill_name": self.skill.name,
        "instructions": self.skill.instructions,
    }

    resource_id = args.get("resource_id")
    if resource_id:
      resource_content = (
          self.skill.resources.get_reference(resource_id)
          or self.skill.resources.get_asset(resource_id)
      )
      script = self.skill.resources.get_script(resource_id)
      if script:
        resource_content = str(script)

      if resource_content:
        result["resource"] = resource_content
      else:
        result["resource_error"] = (
            f"Resource '{resource_id}' not found. "
            f"Available references: {self.skill.resources.list_references()}, "
            f"assets: {self.skill.resources.list_assets()}, "
            f"scripts: {self.skill.resources.list_scripts()}"
        )

    # Include available resource IDs in the response for discoverability
    if (
        self.skill.resources.references
        or self.skill.resources.assets
        or self.skill.resources.scripts
    ):
      result["available_resources"] = {
          "references": self.skill.resources.list_references(),
          "assets": self.skill.resources.list_assets(),
          "scripts": self.skill.resources.list_scripts(),
      }

    return result


def _skill_name_to_tool_name(skill_name: str) -> str:
  """Converts a skill name (kebab-case) to a valid tool name (snake_case).

  Args:
    skill_name: Skill name in kebab-case.

  Returns:
    Valid tool name in snake_case.
  """
  return skill_name.replace("-", "_")

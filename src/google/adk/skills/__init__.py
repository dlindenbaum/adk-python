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

"""Agent Development Kit - Skills."""

from .loader import load_skill
from .loader import load_skill_from_content
from .loader import load_skill_from_dict
from .loader import load_skills
from .models import Frontmatter
from .models import Resources
from .models import Script
from .models import Skill
from .prompt import format_skills_as_xml
from .skills_tool import SkillsTool
from .skills_toolset import SkillsToolset

__all__ = [
    "Frontmatter",
    "Resources",
    "Script",
    "Skill",
    "SkillsTool",
    "SkillsToolset",
    "format_skills_as_xml",
    "load_skill",
    "load_skill_from_content",
    "load_skill_from_dict",
    "load_skills",
]

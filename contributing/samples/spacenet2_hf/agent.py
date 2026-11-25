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

"""SpaceNet 2 building-footprint agent backed by Hugging Face.

Both halves run on Hugging Face:

* Vision: a dedicated Inference Endpoint serving `endpoint/handler.py`
  (Mask R-CNN from `benjbritton/spacenet2-building-detector`), addressed by
  `SPACENET2_ENDPOINT_URL`.
* Reasoning: an LLM on Hugging Face Inference Providers through the
  OpenAI-compatible router, chosen with `SPACENET2_AGENT_MODEL`.

Both read the `HF_TOKEN` environment variable.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from . import spacenet2

HF_ROUTER_URL = 'https://router.huggingface.co/v1'
DEFAULT_AGENT_MODEL = 'openai/gpt-oss-120b'
# Each tile is a 2.5 MB GeoTIFF plus an endpoint call, so tool calls that
# fan out over tiles are capped to keep a single turn responsive.
MAX_TILES_PER_CALL = 200

_client = spacenet2.EndpointClient()


def _summary(result: spacenet2.TileResult) -> dict[str, Any]:
  buildings = result.detection['buildings']
  areas = [b['area_px'] for b in buildings]
  summary = {
      'tile_id': result.tile_id,
      'num_buildings': len(buildings),
      'score_threshold': result.detection['score_threshold'],
      'median_area_m2': (
          # PS-RGB pixels are 0.3 m, so one pixel is 0.09 m^2.
          round(sorted(areas)[len(areas) // 2] * 0.09, 1)
          if areas
          else None
      ),
      'top_scores': [b['score'] for b in buildings[:5]],
  }
  if result.counts is not None:
    summary['spacenet_metric'] = result.counts.as_dict()
  return summary


async def list_spacenet2_tiles(
    city: str, split: str = 'val', limit: int = 10
) -> dict[str, Any]:
  """Lists SpaceNet 2 tile ids.

  Args:
    city: One of AOI_2_Vegas, AOI_3_Paris, AOI_4_Shanghai, AOI_5_Khartoum.
    split: 'val' (labelled, unseen by the model), 'train' (labelled, seen in
      training) or 'test' (public test set, no labels).
    limit: Maximum number of ids to return.

  Returns:
    The tile ids, in a reproducible shuffled order.
  """
  tiles = await asyncio.to_thread(spacenet2.list_tiles, city, split, limit)
  return {'city': city, 'split': split, 'tile_ids': tiles}


async def detect_buildings(
    tile_id: str,
    tool_context: ToolContext,
    score_threshold: float = spacenet2.DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
  """Detects building footprints on one tile with the Inference Endpoint.

  Saves an overlay PNG and a lon/lat GeoJSON of the footprints as artifacts.
  For labelled tiles the result also includes the SpaceNet metric.

  Args:
    tile_id: A tile id such as 'AOI_2_Vegas_img29' or 'test:AOI_3_Paris_img1'.
    score_threshold: Minimum detection confidence; 0.544 is the operating
      point the model was tuned for.

  Returns:
    Building count, area and score summary, metric and artifact names.
  """
  result = await asyncio.to_thread(
      spacenet2.detect_tile,
      _client,
      tile_id,
      score_threshold,
      True,
  )
  stem = result.tile_id.replace(':', '_')
  overlay = base64.b64decode(result.detection.pop('overlay_png'))
  await tool_context.save_artifact(
      f'{stem}_overlay.png',
      types.Part.from_bytes(data=overlay, mime_type='image/png'),
  )
  geojson = json.dumps(spacenet2.to_geojson(result.tile_id, result.detection))
  await tool_context.save_artifact(
      f'{stem}_buildings.geojson',
      types.Part.from_bytes(
          data=geojson.encode(), mime_type='application/geo+json'
      ),
  )
  summary = _summary(result)
  summary['artifacts'] = [f'{stem}_overlay.png', f'{stem}_buildings.geojson']
  return summary


async def evaluate_spacenet2(
    cities: list[str],
    tiles_per_city: int = 10,
    score_threshold: float = spacenet2.DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
  """Scores the endpoint on held-out SpaceNet 2 tiles with the SpaceNet F1.

  A detection is a true positive when its polygon IoU with an unmatched label
  is at least 0.5. Counts are pooled per city; the headline is the macro
  average of the city F1 scores, as in the SpaceNet 2 challenge.

  Args:
    cities: Cities to score, e.g. ['AOI_2_Vegas', 'AOI_5_Khartoum'].
    tiles_per_city: Validation tiles to sample per city.
    score_threshold: Minimum detection confidence.

  Returns:
    Precision, recall and F1 per city, and the macro F1.
  """
  per_city = min(tiles_per_city, MAX_TILES_PER_CALL // max(len(cities), 1))
  tile_ids = [
      t
      for city in cities
      for t in await asyncio.to_thread(
          spacenet2.list_tiles, city, 'val', per_city
      )
  ]
  return await asyncio.to_thread(
      spacenet2.evaluate, _client, tile_ids, score_threshold
  )


async def tune_score_threshold(
    cities: list[str],
    tiles_per_city: int = 10,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
  """Finds the score threshold with the best macro SpaceNet F1.

  Runs the endpoint once per tile at the lowest threshold, then rescores at
  every threshold locally.

  Args:
    cities: Cities to tune on.
    tiles_per_city: Validation tiles to sample per city.
    thresholds: Candidate thresholds; defaults to 0.3 to 0.8 in 0.05 steps.

  Returns:
    Macro and per-city F1 for each threshold, and the best row.
  """
  thresholds = thresholds or [round(0.3 + 0.05 * i, 2) for i in range(11)]
  per_city = min(tiles_per_city, MAX_TILES_PER_CALL // max(len(cities), 1))
  tile_ids = [
      t
      for city in cities
      for t in await asyncio.to_thread(
          spacenet2.list_tiles, city, 'val', per_city
      )
  ]
  return await asyncio.to_thread(
      spacenet2.threshold_sweep, _client, tile_ids, thresholds
  )


async def export_submission(
    tile_ids: list[str],
    tool_context: ToolContext,
    score_threshold: float = spacenet2.DEFAULT_SCORE_THRESHOLD,
) -> dict[str, Any]:
  """Writes a SpaceNet 2 submission CSV for the given tiles as an artifact.

  Args:
    tile_ids: Tiles to include, typically from list_spacenet2_tiles with
      split='test'.
    score_threshold: Minimum detection confidence.

  Returns:
    The artifact name and row counts.
  """
  tile_ids = tile_ids[:MAX_TILES_PER_CALL]

  def run() -> list[tuple[str, int, str, float]]:
    rows = []
    for tile_id in tile_ids:
      result = spacenet2.detect_tile(_client, tile_id, score_threshold)
      rows += spacenet2.submission_rows(result.tile_id, result.detection)
    return rows

  rows = await asyncio.to_thread(run)
  buf = io.StringIO()
  writer = csv.writer(buf)
  writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
  writer.writerows(rows)
  name = 'spacenet2_submission.csv'
  await tool_context.save_artifact(
      name,
      types.Part.from_bytes(data=buf.getvalue().encode(), mime_type='text/csv'),
  )
  buildings = sum(1 for r in rows if r[1] >= 0)
  return {'artifact': name, 'tiles': len(tile_ids), 'buildings': buildings}


def _agent_model() -> LiteLlm:
  return LiteLlm(
      model='openai/'
      + os.environ.get('SPACENET2_AGENT_MODEL', DEFAULT_AGENT_MODEL),
      api_base=HF_ROUTER_URL,
      api_key=os.environ.get('HF_TOKEN'),
  )


root_agent = Agent(
    name='spacenet2_agent',
    model=_agent_model(),
    description=(
        'Extracts building footprints from SpaceNet 2 satellite tiles with a'
        ' Hugging Face Inference Endpoint and scores them.'
    ),
    instruction="""\
You are a remote-sensing analyst working on SpaceNet 2: building footprint
extraction from 30 cm WorldView-3 imagery of Las Vegas (AOI_2_Vegas), Paris
(AOI_3_Paris), Shanghai (AOI_4_Shanghai) and Khartoum (AOI_5_Khartoum).

The detector is a Mask R-CNN behind a Hugging Face Inference Endpoint. Use the
tools rather than guessing:
- list_spacenet2_tiles to find tiles. 'val' tiles are labelled and were not
  used in training, so use them for any claim about accuracy. Never report
  accuracy measured on 'train' tiles as model performance.
- detect_buildings for a single tile. Point the user to the overlay artifact.
- evaluate_spacenet2 for the SpaceNet F1 (IoU >= 0.5, macro over cities).
- tune_score_threshold to choose an operating point, then confirm it with
  evaluate_spacenet2 on a different, larger sample before recommending it.
- export_submission to write a SpaceNet-format CSV, usually for 'test' tiles.

Report numbers exactly as the tools return them and state how many tiles they
rest on; small samples are noisy, so say so.
""",
    tools=[
        list_spacenet2_tiles,
        detect_buildings,
        evaluate_spacenet2,
        tune_score_threshold,
        export_submission,
    ],
)

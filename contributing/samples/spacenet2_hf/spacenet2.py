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

"""SpaceNet 2 data access, endpoint client and the SpaceNet F1 metric.

Tiles and labels are read straight from the public SpaceNet bucket on AWS
(`s3://spacenet-dataset`, anonymous HTTPS). Inference runs on a Hugging Face
Inference Endpoint that serves `endpoint/handler.py`.

Tile ids follow the SpaceNet ImageId convention, e.g. `AOI_2_Vegas_img29`.
Prefix `test:` for the public test set, which has no labels.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from collections.abc import Sequence
import concurrent.futures
import dataclasses
import functools
import json
import os
import pathlib
import random
import re
import time
from typing import Any

import httpx
from shapely import geometry
from shapely import wkt

CITIES = ('AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum')
S3_BASE = 'https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings'
# The 80/20 split the model was trained with; only `val` tiles are unseen.
SPLIT_URL = (
    'https://raw.githubusercontent.com/benjbritton/geoai-detection/'
    'master/configs/spacenet2_split.json'
)
DEFAULT_ENDPOINT_URL = 'http://127.0.0.1:8080'
DEFAULT_SCORE_THRESHOLD = 0.544
IOU_THRESHOLD = 0.5
_TEST_PREFIX = 'test:'
_TILE_ID_RE = re.compile(r'^(AOI_\d_[A-Za-z]+)_(img\d+)$')
_CACHE_DIR = pathlib.Path(
    os.environ.get(
        'SPACENET2_CACHE_DIR',
        pathlib.Path.home() / '.cache' / 'adk_spacenet2_hf',
    )
)


@dataclasses.dataclass(frozen=True)
class Tile:
  """One SpaceNet 2 tile."""

  city: str
  image: str
  test: bool = False

  @classmethod
  def parse(cls, tile_id: str) -> Tile:
    test = tile_id.startswith(_TEST_PREFIX)
    match = _TILE_ID_RE.match(tile_id.removeprefix(_TEST_PREFIX).strip())
    if not match or match.group(1) not in CITIES:
      raise ValueError(
          f'Bad tile id {tile_id!r}; expected e.g. "AOI_2_Vegas_img29" or'
          ' "test:AOI_3_Paris_img1".'
      )
    return cls(match.group(1), match.group(2), test)

  @property
  def image_id(self) -> str:
    """SpaceNet ImageId, as used in submission files."""
    return f'{self.city}_{self.image}'

  @property
  def tile_id(self) -> str:
    return (_TEST_PREFIX if self.test else '') + self.image_id

  @property
  def image_url(self) -> str:
    if self.test:
      return (
          f'{S3_BASE}/test_public/{self.city}/PS-RGB/'
          f'SN2_buildings_test_public_{self.city}_PS-RGB_{self.image}.tif'
      )
    return (
        f'{S3_BASE}/train/{self.city}/PS-RGB/'
        f'SN2_buildings_train_{self.city}_PS-RGB_{self.image}.tif'
    )

  @property
  def labels_url(self) -> str | None:
    if self.test:
      return None
    return (
        f'{S3_BASE}/train/{self.city}/geojson_buildings/'
        f'SN2_buildings_train_{self.city}_geojson_buildings_{self.image}.geojson'
    )


def _cached_get(url: str) -> bytes:
  path = _CACHE_DIR / url.removeprefix('https://').replace('/', '__')
  if path.exists():
    return path.read_bytes()
  response = _http_get_with_retries(url)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix('.part')
  tmp.write_bytes(response)
  tmp.replace(path)
  return response


def _http_get_with_retries(url: str, attempts: int = 4) -> bytes:
  for attempt in range(attempts):
    try:
      response = httpx.get(url, timeout=60, follow_redirects=True)
      response.raise_for_status()
      return response.content
    except (httpx.TransportError, httpx.HTTPStatusError):
      if attempt == attempts - 1:
        raise
      time.sleep(2**attempt)
  raise AssertionError('unreachable')


def fetch_image(tile: Tile) -> bytes:
  """Raw 16-bit PS-RGB GeoTIFF bytes."""
  return _cached_get(tile.image_url)


def fetch_labels(tile: Tile) -> list[geometry.Polygon]:
  """Ground-truth footprints in lon/lat (CRS84)."""
  if tile.labels_url is None:
    raise ValueError(f'{tile.tile_id} is a test tile and has no labels.')
  collection = json.loads(_cached_get(tile.labels_url))
  polygons = []
  for feature in collection.get('features', []):
    shape = geometry.shape(feature['geometry'])
    parts = shape.geoms if shape.geom_type == 'MultiPolygon' else [shape]
    for part in parts:
      if part.geom_type == 'Polygon' and not part.is_empty:
        polygons.append(geometry.Polygon([c[:2] for c in part.exterior.coords]))
  return polygons


@functools.lru_cache(maxsize=1)
def load_split() -> dict[str, dict[str, list[str]]]:
  """{city: {'train': [...], 'val': [...]}} from the model's repository."""
  cities = json.loads(_cached_get(SPLIT_URL))['cities']
  return {
      city: {'train': v['train'], 'val': v['val']} for city, v in cities.items()
  }


def list_tiles(
    city: str, split: str = 'val', limit: int = 20, seed: int = 0
) -> list[str]:
  """Tile ids from one city's `val`, `train` or `test` split."""
  if city not in CITIES:
    raise ValueError(f'Unknown city {city!r}; choose from {CITIES}.')
  if split == 'test':
    images = _list_test_images(city)
  elif split in ('train', 'val'):
    images = list(load_split()[city][split])
  else:
    raise ValueError("split must be 'val', 'train' or 'test'.")
  # A seeded shuffle gives a reproducible, unbiased sample of a split.
  random.Random(seed).shuffle(images)
  prefix = _TEST_PREFIX if split == 'test' else ''
  return [f'{prefix}{city}_{image}' for image in images[:limit]]


@functools.lru_cache(maxsize=len(CITIES))
def _list_test_images(city: str) -> list[str]:
  prefix = f'spacenet/SN2_buildings/test_public/{city}/PS-RGB/'
  images, token = [], None
  while True:
    params = {'list-type': '2', 'prefix': prefix}
    if token:
      params['continuation-token'] = token
    response = httpx.get(
        'https://spacenet-dataset.s3.amazonaws.com/', params=params, timeout=60
    )
    response.raise_for_status()
    images += re.findall(r'_PS-RGB_(img\d+)\.tif</Key>', response.text)
    match = re.search(
        r'<NextContinuationToken>([^<]+)</NextContinuationToken>',
        response.text,
    )
    if not match:
      return sorted(images, key=lambda s: int(s[3:]))
    token = match.group(1)


class EndpointClient:
  """Client for the Inference Endpoint that serves `endpoint/handler.py`."""

  def __init__(
      self,
      url: str | None = None,
      token: str | None = None,
      timeout: float = 120,
      cold_start_wait: float = 600,
  ):
    self.url = url or os.environ.get(
        'SPACENET2_ENDPOINT_URL', DEFAULT_ENDPOINT_URL
    )
    self.token = token or os.environ.get('HF_TOKEN')
    self.timeout = timeout
    # Scale-to-zero endpoints answer 503 while a replica boots.
    self.cold_start_wait = cold_start_wait

  def detect(
      self,
      image: bytes,
      score_threshold: float = DEFAULT_SCORE_THRESHOLD,
      return_overlay: bool = False,
  ) -> dict[str, Any]:
    headers = {'Content-Type': 'application/json'}
    if self.token:
      headers['Authorization'] = f'Bearer {self.token}'
    body = {
        'inputs': base64.b64encode(image).decode(),
        'parameters': {
            'score_threshold': score_threshold,
            'return_overlay': return_overlay,
        },
    }
    deadline = time.monotonic() + self.cold_start_wait
    delay = 5.0
    while True:
      response = httpx.post(
          self.url, json=body, headers=headers, timeout=self.timeout
      )
      if response.status_code != 503 or time.monotonic() > deadline:
        response.raise_for_status()
        return response.json()
      time.sleep(delay)
      delay = min(delay * 2, 60)


def pixel_polygons(
    lonlat: Iterable[geometry.Polygon], transform: Sequence[float]
) -> list[geometry.Polygon]:
  """Maps polygons through the inverse of a GDAL-style affine transform."""
  a, b, c, d, e, f = transform
  det = a * e - b * d
  polygons = []
  for polygon in lonlat:
    ring = []
    for x, y in polygon.exterior.coords:
      dx, dy = x - c, y - f
      ring.append(((e * dx - b * dy) / det, (a * dy - d * dx) / det))
    polygons.append(geometry.Polygon(ring))
  return polygons


def lonlat_polygon(
    ring: Sequence[Sequence[float]], transform: Sequence[float]
) -> geometry.Polygon:
  a, b, c, d, e, f = transform
  return geometry.Polygon(
      [(a * x + b * y + c, d * x + e * y + f) for x, y in ring]
  )


@dataclasses.dataclass
class Counts:
  """True/false positives and false negatives under the SpaceNet metric."""

  tp: int = 0
  fp: int = 0
  fn: int = 0

  def __iadd__(self, other: Counts) -> Counts:
    self.tp += other.tp
    self.fp += other.fp
    self.fn += other.fn
    return self

  @property
  def precision(self) -> float:
    return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

  @property
  def recall(self) -> float:
    return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

  @property
  def f1(self) -> float:
    p, r = self.precision, self.recall
    return 2 * p * r / (p + r) if p + r else 0.0

  def as_dict(self) -> dict[str, float]:
    return {
        'tp': self.tp,
        'fp': self.fp,
        'fn': self.fn,
        'precision': round(self.precision, 4),
        'recall': round(self.recall, 4),
        'f1': round(self.f1, 4),
    }


def _valid(polygon: geometry.Polygon) -> geometry.Polygon:
  return polygon if polygon.is_valid else polygon.buffer(0)


def spacenet_match(
    proposals: Sequence[geometry.Polygon],
    truth: Sequence[geometry.Polygon],
    iou_threshold: float = IOU_THRESHOLD,
) -> Counts:
  """SpaceNet matching: each proposal, by falling score, claims the unmatched
  label with the highest IoU if that IoU reaches the threshold."""
  remaining = [_valid(t) for t in truth]
  counts = Counts()
  for proposal in (_valid(p) for p in proposals):
    best_iou, best = 0.0, -1
    for i, label in enumerate(remaining):
      if not proposal.intersects(label):
        continue
      union = proposal.union(label).area
      iou = proposal.intersection(label).area / union if union else 0.0
      if iou > best_iou:
        best_iou, best = iou, i
    if best_iou >= iou_threshold:
      counts.tp += 1
      remaining.pop(best)
    else:
      counts.fp += 1
  counts.fn = len(remaining)
  return counts


@dataclasses.dataclass
class TileResult:
  tile_id: str
  detection: dict[str, Any]
  counts: Counts | None


def detect_tile(
    client: EndpointClient,
    tile_id: str,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    return_overlay: bool = False,
) -> TileResult:
  """Runs the endpoint on one tile and scores it when labels exist."""
  tile = Tile.parse(tile_id)
  detection = client.detect(
      fetch_image(tile), score_threshold, return_overlay=return_overlay
  )
  counts = None
  if not tile.test:
    georef = detection.get('georef')
    if not georef:
      raise ValueError(f'Endpoint returned no georef for {tile_id}.')
    truth = pixel_polygons(fetch_labels(tile), georef['transform'])
    proposals = [geometry.Polygon(b['polygon']) for b in detection['buildings']]
    counts = spacenet_match(proposals, truth)
  return TileResult(tile.tile_id, detection, counts)


def evaluate(
    client: EndpointClient,
    tile_ids: Sequence[str],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    workers: int = 4,
    on_result=None,
) -> dict[str, Any]:
  """SpaceNet F1 per city (counts pooled over tiles) and the city macro."""
  per_city: dict[str, Counts] = {}
  errors = []
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    futures = {
        pool.submit(detect_tile, client, t, score_threshold): t
        for t in tile_ids
    }
    for future in concurrent.futures.as_completed(futures):
      tile_id = futures[future]
      try:
        result = future.result()
      except (httpx.HTTPError, ValueError) as e:
        errors.append(f'{tile_id}: {e!r}')
        continue
      if result.counts is None:
        continue
      city = Tile.parse(tile_id).city
      per_city.setdefault(city, Counts())
      per_city[city] += result.counts
      if on_result:
        on_result(result)
  cities = {city: c.as_dict() for city, c in sorted(per_city.items())}
  macro = (
      sum(c.f1 for c in per_city.values()) / len(per_city) if per_city else 0.0
  )
  return {
      'score_threshold': score_threshold,
      'tiles_scored': len(tile_ids) - len(errors),
      'cities': cities,
      'macro_f1': round(macro, 4),
      'errors': errors,
  }


def threshold_sweep(
    client: EndpointClient,
    tile_ids: Sequence[str],
    thresholds: Sequence[float],
    workers: int = 4,
) -> dict[str, Any]:
  """Macro F1 at each score threshold, from one endpoint pass per tile."""
  floor = min(thresholds)
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    results = list(
        pool.map(lambda t: _proposals_and_truth(client, t, floor), tile_ids)
    )
  table = []
  for threshold in sorted(thresholds):
    per_city: dict[str, Counts] = {}
    for city, scored, truth in results:
      kept = [p for s, p in scored if s >= threshold]
      per_city.setdefault(city, Counts())
      per_city[city] += spacenet_match(kept, truth)
    macro = sum(c.f1 for c in per_city.values()) / len(per_city)
    table.append({
        'score_threshold': threshold,
        'macro_f1': round(macro, 4),
        'f1_by_city': {c: round(v.f1, 4) for c, v in sorted(per_city.items())},
    })
  best = max(table, key=lambda row: row['macro_f1'])
  return {'tiles': len(tile_ids), 'sweep': table, 'best': best}


def _proposals_and_truth(client: EndpointClient, tile_id: str, floor: float):
  tile = Tile.parse(tile_id)
  if tile.test:
    raise ValueError('Threshold sweeps need labelled (val/train) tiles.')
  detection = client.detect(fetch_image(tile), floor)
  truth = pixel_polygons(fetch_labels(tile), detection['georef']['transform'])
  scored = [
      (b['score'], geometry.Polygon(b['polygon']))
      for b in detection['buildings']
  ]
  return tile.city, scored, truth


def submission_rows(
    tile_id: str, detection: dict[str, Any]
) -> list[tuple[str, int, str, float]]:
  """Rows in the SpaceNet submission format (pixel-space WKT)."""
  image_id = Tile.parse(tile_id).image_id
  rows = [
      (
          image_id,
          i,
          wkt.dumps(geometry.Polygon(b['polygon']), rounding_precision=2),
          b['score'],
      )
      for i, b in enumerate(detection['buildings'])
  ]
  # SpaceNet marks a tile with no buildings by a single empty row.
  return rows or [(image_id, -1, 'POLYGON EMPTY', 1.0)]


def to_geojson(tile_id: str, detection: dict[str, Any]) -> dict[str, Any]:
  """Detections as a lon/lat FeatureCollection (needs a georeferenced tile)."""
  georef = detection.get('georef')
  if not georef:
    raise ValueError(f'{tile_id} has no georeferencing.')
  features = [
      {
          'type': 'Feature',
          'properties': {'score': b['score'], 'tile_id': tile_id},
          'geometry': geometry.mapping(
              lonlat_polygon(b['polygon'], georef['transform'])
          ),
      }
      for b in detection['buildings']
  ]
  return {'type': 'FeatureCollection', 'features': features}

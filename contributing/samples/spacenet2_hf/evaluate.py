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

"""Batch SpaceNet 2 scoring or submission export against the endpoint.

# SpaceNet F1 on 100 held-out tiles per city.
python evaluate.py --per-city 100 --out results.json

# Pixel-space submission CSV for the public test set.
python evaluate.py --split test --per-city 0 --submission test.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spacenet2  # pylint: disable=g-import-not-at-top


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--split', default='val', choices=['val', 'test'])
  parser.add_argument(
      '--per-city', type=int, default=50, help='tiles per city; 0 means all'
  )
  parser.add_argument('--cities', nargs='+', default=list(spacenet2.CITIES))
  parser.add_argument(
      '--threshold', type=float, default=spacenet2.DEFAULT_SCORE_THRESHOLD
  )
  parser.add_argument('--workers', type=int, default=4)
  parser.add_argument('--endpoint-url')
  parser.add_argument('--out', help='write the metrics JSON here')
  parser.add_argument('--submission', help='write a SpaceNet CSV here')
  args = parser.parse_args()

  client = spacenet2.EndpointClient(url=args.endpoint_url)
  limit = args.per_city or 10**6
  tile_ids = [
      t
      for city in args.cities
      for t in spacenet2.list_tiles(city, args.split, limit=limit)
  ]
  print(f'{len(tile_ids)} {args.split} tiles -> {client.url}', flush=True)
  start = time.monotonic()

  if args.split == 'val':
    done = [0]

    def progress(_):
      done[0] += 1
      if done[0] % 25 == 0:
        rate = done[0] / (time.monotonic() - start)
        print(f'  {done[0]}/{len(tile_ids)} ({rate:.2f} tiles/s)', flush=True)

    report = spacenet2.evaluate(
        client, tile_ids, args.threshold, args.workers, on_result=progress
    )
    report['seconds'] = round(time.monotonic() - start, 1)
    print(json.dumps(report, indent=2))
    if args.out:
      with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    return

  with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
    results = list(
        pool.map(
            lambda t: spacenet2.detect_tile(client, t, args.threshold),
            tile_ids,
        )
    )
  rows = [
      r
      for res in results
      for r in spacenet2.submission_rows(res.tile_id, res.detection)
  ]
  path = args.submission or 'spacenet2_submission.csv'
  with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
    writer.writerows(rows)
  print(f'{len(rows)} rows for {len(results)} tiles -> {path}')


if __name__ == '__main__':
  main()

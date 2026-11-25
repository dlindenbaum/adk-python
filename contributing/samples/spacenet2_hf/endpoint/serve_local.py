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

"""Serves `handler.py` locally with the Inference Endpoints HTTP contract.

Useful for development before paying for a dedicated endpoint:

  python serve_local.py --model-dir ./spacenet2-building-detector --port 8080

`--model-dir` is a local snapshot of the model repository; when omitted it is
downloaded from the Hub.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import EndpointHandler  # pylint: disable=g-import-not-at-top
from handler import MODEL_ID


def create_app(model_dir: str) -> FastAPI:
  handler = EndpointHandler(model_dir)
  app = FastAPI(title='SpaceNet 2 building detector (local endpoint)')

  @app.get('/health')
  async def health():
    return {'status': 'ok', 'model': MODEL_ID}

  @app.post('/')
  async def predict(request: Request):
    content_type = request.headers.get('content-type', '')
    if content_type.startswith('application/json'):
      data = await request.json()
    else:
      data = {'inputs': await request.body()}
    try:
      result = await asyncio.to_thread(handler, data)
    except (KeyError, ValueError, OSError) as e:
      return JSONResponse({'error': repr(e)}, status_code=400)
    return result

  return app


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--model-dir')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', type=int, default=8080)
  args = parser.parse_args()
  model_dir = args.model_dir
  if not model_dir:
    from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top

    model_dir = snapshot_download(
        MODEL_ID, allow_patterns=['model_final.pth', 'config.yaml']
    )
  uvicorn.run(create_app(model_dir), host=args.host, port=args.port)


if __name__ == '__main__':
  main()

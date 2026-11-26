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

"""Publishes the handler and creates a dedicated HF Inference Endpoint.

Needs an `HF_TOKEN` with write access and a billing method on the account or
organization (`--namespace`). Steps:

1. Copy `model_final.pth` and `config.yaml` from the source model into a model
   repo of yours, next to `handler.py` and `requirements.txt`.
2. Create an authenticated, scale-to-zero endpoint with `task="custom"`, so the
   Inference Toolkit loads `handler.EndpointHandler`.

  python deploy_endpoint.py --dry-run
  python deploy_endpoint.py --accelerator gpu --instance-type nvidia-t4
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

SOURCE_MODEL = 'benjbritton/spacenet2-building-detector'
HERE = os.path.dirname(os.path.abspath(__file__))

_CARD = f"""---
license: cc-by-sa-4.0
base_model: {SOURCE_MODEL}
pipeline_tag: image-segmentation
tags: [spacenet, building-detection, remote-sensing, inference-endpoints]
---

# SpaceNet 2 building detector: Inference Endpoints package

The weights of [{SOURCE_MODEL}](https://huggingface.co/{SOURCE_MODEL}) by
Benjamin J. Britton (CC BY-SA 4.0), with a custom `handler.py` that serves
them on plain torchvision. Trained on SpaceNet 2 imagery, which SpaceNet
Partners license under CC BY-SA 4.0.
"""


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--namespace', help='user or org; default: token owner')
  parser.add_argument('--repo-name', default='spacenet2-building-endpoint')
  parser.add_argument('--endpoint-name', default='spacenet2-buildings')
  parser.add_argument('--accelerator', default='gpu', choices=['gpu', 'cpu'])
  parser.add_argument('--instance-type', default='nvidia-t4')
  parser.add_argument('--instance-size', default='x1')
  parser.add_argument('--vendor', default='aws')
  parser.add_argument('--region', default='us-east-1')
  parser.add_argument('--scale-to-zero-minutes', type=int, default=15)
  parser.add_argument('--dry-run', action='store_true')
  args = parser.parse_args()

  api = HfApi()
  namespace = args.namespace or api.whoami()['name']
  repo_id = f'{namespace}/{args.repo_name}'
  endpoint_kwargs = dict(
      repository=repo_id,
      framework='pytorch',
      task='custom',
      accelerator=args.accelerator,
      instance_type=args.instance_type,
      instance_size=args.instance_size,
      vendor=args.vendor,
      region=args.region,
      min_replica=0,
      max_replica=1,
      scale_to_zero_timeout=args.scale_to_zero_minutes,
      type='authenticated',
      namespace=namespace,
  )
  print(f'model repo: {repo_id} (private)')
  print(f'endpoint:   {args.endpoint_name} {endpoint_kwargs}')
  if args.dry_run:
    return

  try:
    with tempfile.TemporaryDirectory() as staging:
      for name in ('model_final.pth', 'config.yaml'):
        shutil.copy(hf_hub_download(SOURCE_MODEL, name), staging)
      for name in ('handler.py', 'requirements.txt'):
        shutil.copy(os.path.join(HERE, name), staging)
      with open(os.path.join(staging, 'README.md'), 'w') as f:
        f.write(_CARD)
      api.create_repo(repo_id, private=True, exist_ok=True)
      api.upload_folder(
          repo_id=repo_id,
          folder_path=staging,
          commit_message='Inference Endpoints handler for SpaceNet 2',
      )
    endpoint = api.create_inference_endpoint(
        args.endpoint_name, **endpoint_kwargs
    )
  except HfHubHTTPError as e:
    raise SystemExit(
        f'Hub refused the request: {e}\nCreating endpoints needs a write'
        ' token and a payment method on the namespace.'
    ) from e
  endpoint.wait(timeout=1800)
  print(f'ready: export SPACENET2_ENDPOINT_URL={endpoint.url}')


if __name__ == '__main__':
  main()

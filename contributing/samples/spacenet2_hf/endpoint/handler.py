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

"""Hugging Face Inference Endpoints handler for SpaceNet 2 building footprints.

Serves `benjbritton/spacenet2-building-detector` (a Detectron2 Mask R-CNN
R50-FPN checkpoint) with plain torchvision, so the endpoint container needs no
compiled Detectron2. The Detectron2 graph is reproduced exactly where it
differs from torchvision's defaults:

* MSRA-style ResNet: the stride of each stage sits on the 1x1 conv.
* ROIAlignV2 pooling (`aligned=True`) with Detectron2's level assignment.
* Unrounded cell anchors.
* Detectron2 puts the background class last; torchvision puts it first.
* BGR input in [0, 255] with Detectron2's pixel mean and unit std.

Request (JSON):
  {"inputs": "<base64 PNG/JPEG/GeoTIFF>",
   "parameters": {"score_threshold": 0.544, "return_overlay": false}}

16-bit GeoTIFFs (SpaceNet PS-RGB) are stretched per band between the 2nd and
98th percentile of nonzero pixels, as in training. For a georeferenced input
the response also carries the affine transform and CRS so callers can map
pixel polygons to the ground.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely import geometry
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import roi_align
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

MODEL_ID = 'benjbritton/spacenet2-building-detector'
DEFAULT_SCORE_THRESHOLD = 0.544
# Detectron2 keeps detections down to the operating threshold; requests may
# lower it, so the network itself runs with a permissive floor.
_MIN_SCORE_THRESHOLD = 0.05
_PIXEL_MEAN_BGR = (103.530, 116.280, 123.675)
_FPN_STRIDES = (4, 8, 16, 32)
_CANONICAL_LEVEL = 4
_CANONICAL_BOX_SIZE = 224


class _D2AnchorGenerator(AnchorGenerator):
  """AnchorGenerator whose cell anchors match Detectron2 (no rounding)."""

  def generate_anchors(
      self,
      scales,
      aspect_ratios,
      dtype: torch.dtype = torch.float32,
      device: torch.device = torch.device('cpu'),
  ) -> torch.Tensor:
    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    # Aspect-major order, as in torchvision; with one size per level this is
    # also Detectron2's order.
    ws = torch.sqrt(scales[None, :] ** 2 / ratios[:, None])
    hs = ratios[:, None] * ws
    ws, hs = ws.reshape(-1), hs.reshape(-1)
    return torch.stack([-ws, -hs, ws, hs], dim=1) / 2


class _D2RoIPooler(nn.Module):
  """Detectron2's ROIPooler with ROIAlignV2 over FPN levels p2-p5."""

  def __init__(self, output_size: int):
    super().__init__()
    self.output_size = output_size

  def forward(self, features, proposals, image_shapes):
    del image_shapes  # Scales are fixed by the FPN strides.
    levels = [features[str(i)] for i in range(len(_FPN_STRIDES))]
    boxes = torch.cat(proposals, dim=0)
    batch_idx = torch.cat([
        torch.full((len(b), 1), i, dtype=boxes.dtype, device=boxes.device)
        for i, b in enumerate(proposals)
    ])
    rois = torch.cat([batch_idx, boxes], dim=1)
    box_sizes = torch.sqrt(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    )
    level = torch.floor(
        _CANONICAL_LEVEL + torch.log2(box_sizes / _CANONICAL_BOX_SIZE + 1e-8)
    )
    level = level.clamp(2, 5).to(torch.int64) - 2
    out = levels[0].new_zeros(
        (len(rois), levels[0].shape[1], self.output_size, self.output_size)
    )
    for i, (feature, stride) in enumerate(zip(levels, _FPN_STRIDES)):
      idx = torch.nonzero(level == i).squeeze(1)
      if idx.numel():
        out[idx] = roi_align(
            feature,
            rois[idx],
            self.output_size,
            spatial_scale=1.0 / stride,
            sampling_ratio=0,
            aligned=True,
        )
    return out


def build_model() -> MaskRCNN:
  """Builds the torchvision graph equivalent to the Detectron2 config."""
  body = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
  for stage in (body.layer2, body.layer3, body.layer4):
    stage[0].conv1.stride = (2, 2)
    stage[0].conv2.stride = (1, 1)
  backbone = BackboneWithFPN(
      body,
      return_layers={
          'layer1': '0',
          'layer2': '1',
          'layer3': '2',
          'layer4': '3',
      },
      in_channels_list=[256, 512, 1024, 2048],
      out_channels=256,
      extra_blocks=LastLevelMaxPool(),
  )
  anchors = _D2AnchorGenerator(
      sizes=((32,), (64,), (128,), (256,), (512,)),
      aspect_ratios=((0.5, 1.0, 2.0),) * 5,
  )
  model = MaskRCNN(
      backbone,
      num_classes=2,
      min_size=800,
      max_size=1333,
      image_mean=list(_PIXEL_MEAN_BGR),
      image_std=[1.0, 1.0, 1.0],
      rpn_anchor_generator=anchors,
      rpn_pre_nms_top_n_test=1000,
      rpn_post_nms_top_n_test=1000,
      rpn_nms_thresh=0.7,
      box_score_thresh=_MIN_SCORE_THRESHOLD,
      box_nms_thresh=0.5,
      box_detections_per_img=100,
  )
  # MaskRCNN's constructor only accepts MultiScaleRoIAlign, which pools with
  # aligned=False, so the Detectron2 poolers go in afterwards.
  model.roi_heads.box_roi_pool = _D2RoIPooler(7)
  model.roi_heads.mask_roi_pool = _D2RoIPooler(14)
  return model


def _convert_state_dict(d2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Renames Detectron2 checkpoint tensors to the torchvision graph."""
  out = {}
  bn = ('weight', 'bias', 'running_mean', 'running_var')
  stem = 'backbone.bottom_up.stem.conv1'
  out['backbone.body.conv1.weight'] = d2[f'{stem}.weight']
  for p in bn:
    out[f'backbone.body.bn1.{p}'] = d2[f'{stem}.norm.{p}']
  for stage, n_blocks in ((2, 3), (3, 4), (4, 6), (5, 3)):
    for block in range(n_blocks):
      src = f'backbone.bottom_up.res{stage}.{block}'
      dst = f'backbone.body.layer{stage - 1}.{block}'
      for conv in (1, 2, 3):
        out[f'{dst}.conv{conv}.weight'] = d2[f'{src}.conv{conv}.weight']
        for p in bn:
          out[f'{dst}.bn{conv}.{p}'] = d2[f'{src}.conv{conv}.norm.{p}']
      if block == 0:
        out[f'{dst}.downsample.0.weight'] = d2[f'{src}.shortcut.weight']
        for p in bn:
          out[f'{dst}.downsample.1.{p}'] = d2[f'{src}.shortcut.norm.{p}']
  for i, level in enumerate(range(2, 6)):
    for p in ('weight', 'bias'):
      out[f'backbone.fpn.inner_blocks.{i}.0.{p}'] = d2[
          f'backbone.fpn_lateral{level}.{p}'
      ]
      out[f'backbone.fpn.layer_blocks.{i}.0.{p}'] = d2[
          f'backbone.fpn_output{level}.{p}'
      ]
  rpn = 'proposal_generator.rpn_head'
  for p in ('weight', 'bias'):
    out[f'rpn.head.conv.0.0.{p}'] = d2[f'{rpn}.conv.{p}']
    out[f'rpn.head.cls_logits.{p}'] = d2[f'{rpn}.objectness_logits.{p}']
    out[f'rpn.head.bbox_pred.{p}'] = d2[f'{rpn}.anchor_deltas.{p}']
    out[f'roi_heads.box_head.fc6.{p}'] = d2[f'roi_heads.box_head.fc1.{p}']
    out[f'roi_heads.box_head.fc7.{p}'] = d2[f'roi_heads.box_head.fc2.{p}']
    # Detectron2 scores are [building, background]; torchvision wants
    # [background, building].
    out[f'roi_heads.box_predictor.cls_score.{p}'] = d2[
        f'roi_heads.box_predictor.cls_score.{p}'
    ][[1, 0]]
    fg = d2[f'roi_heads.box_predictor.bbox_pred.{p}']
    out[f'roi_heads.box_predictor.bbox_pred.{p}'] = torch.cat(
        [torch.zeros_like(fg), fg]
    )
    for i in range(4):
      out[f'roi_heads.mask_head.{i}.0.{p}'] = d2[
          f'roi_heads.mask_head.mask_fcn{i + 1}.{p}'
      ]
    out[f'roi_heads.mask_predictor.conv5_mask.{p}'] = d2[
        f'roi_heads.mask_head.deconv.{p}'
    ]
    fg = d2[f'roi_heads.mask_head.predictor.{p}']
    out[f'roi_heads.mask_predictor.mask_fcn_logits.{p}'] = torch.cat(
        [torch.zeros_like(fg), fg]
    )
  return out


def load_model(weights_path: str) -> MaskRCNN:
  """Loads the Detectron2 checkpoint into the torchvision graph."""
  # weights_only=True: the checkpoint is a plain tensor dict, so no pickled
  # code is ever executed.
  checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
  model = build_model()
  model.load_state_dict(_convert_state_dict(checkpoint['model']), strict=True)
  return model.eval()


def percentile_stretch(
    arr: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
  """Stretches each band of (H, W, 3) to uint8, ignoring zero (nodata)."""
  out = np.empty(arr.shape[:2] + (3,), np.uint8)
  for c in range(3):
    band = arr[..., c]
    valid = band[band > 0]
    if valid.size:
      lo = float(np.percentile(valid, low))
      hi = float(np.percentile(valid, high))
    else:
      lo, hi = 0.0, 1.0
    scaled = (band.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)
  return out


def decode_image(
    payload: Any, stretch: bool = False
) -> tuple[np.ndarray, dict[str, Any] | None]:
  """Returns (H, W, 3) uint8 RGB and optional georeferencing."""
  if isinstance(payload, Image.Image):
    rgb = np.asarray(payload.convert('RGB'))
    return (percentile_stretch(rgb) if stretch else rgb), None
  raw = base64.b64decode(payload) if isinstance(payload, str) else payload
  if raw[:4] in (b'II*\x00', b'MM\x00*'):
    import rasterio  # pylint: disable=g-import-not-at-top

    with rasterio.MemoryFile(raw) as mem, mem.open() as src:
      arr = src.read([1, 2, 3]).transpose(1, 2, 0)
      georef = None
      if src.crs is not None:
        t = src.transform
        georef = {
            'crs': src.crs.to_string(),
            'transform': [t.a, t.b, t.c, t.d, t.e, t.f],
        }
  else:
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    georef = None
  if arr.dtype != np.uint8 or stretch:
    arr = percentile_stretch(arr)
  return np.ascontiguousarray(arr), georef


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
  """Outline of the largest blob in continuous pixel coordinates.

  The contour runs through boundary pixel centres; growing it by half a
  pixel puts it on the pixel edges, so polygon IoU against vector labels is
  not biased low for small buildings.
  """
  contours, _ = cv2.findContours(
      mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return []
  ring = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float) + 0.5
  if len(ring) == 1:
    shape = geometry.Point(ring[0])
  elif len(ring) == 2:
    shape = geometry.LineString(ring)
  else:
    shape = geometry.Polygon(ring).buffer(0)
  shape = shape.buffer(0.5, join_style=2)
  if shape.geom_type == 'MultiPolygon':
    shape = max(shape.geoms, key=lambda g: g.area)
  if shape.is_empty or shape.geom_type != 'Polygon':
    return []
  return [[round(x, 2), round(y, 2)] for x, y in shape.exterior.coords[:-1]]


class EndpointHandler:
  """Custom handler contract of Hugging Face Inference Endpoints."""

  def __init__(self, path: str = ''):
    weights = os.path.join(path, 'model_final.pth')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = load_model(weights).to(self.device)
    self._lock = threading.Lock()

  @torch.inference_mode()
  def predict(self, rgb: np.ndarray, score_threshold: float) -> dict[str, Any]:
    bgr = torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1).float()
    with self._lock:
      self.model.roi_heads.score_thresh = min(
          score_threshold, _MIN_SCORE_THRESHOLD
      )
      out = self.model([bgr.to(self.device)])[0]
    keep = out['scores'] >= score_threshold
    scores = out['scores'][keep].cpu().numpy()
    boxes = out['boxes'][keep].cpu().numpy()
    masks = (out['masks'][keep, 0] > 0.5).cpu().numpy()
    buildings = []
    for score, box, mask in sorted(
        zip(scores, boxes, masks), key=lambda t: -t[0]
    ):
      polygon = mask_to_polygon(mask)
      if not polygon:
        continue
      buildings.append({
          'score': round(float(score), 4),
          'bbox': [round(float(v), 1) for v in box],
          'area_px': int(mask.sum()),
          'polygon': polygon,
      })
    return {'buildings': buildings, 'masks': masks}

  def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
    params = data.get('parameters') or {}
    score_threshold = float(
        params.get('score_threshold', DEFAULT_SCORE_THRESHOLD)
    )
    rgb, georef = decode_image(
        data['inputs'], stretch=bool(params.get('stretch', False))
    )
    result = self.predict(rgb, score_threshold)
    response = {
        'model': MODEL_ID,
        'image_size': [int(rgb.shape[0]), int(rgb.shape[1])],
        'score_threshold': score_threshold,
        'num_buildings': len(result['buildings']),
        'buildings': result['buildings'],
        'georef': georef,
    }
    if params.get('return_overlay'):
      overlay = rgb.copy()
      for b in result['buildings']:
        ring = np.round(np.asarray(b['polygon'])).astype(np.int32)
        cv2.polylines(overlay, [ring], True, (255, 220, 0), 2)
      buf = io.BytesIO()
      Image.fromarray(overlay).save(buf, format='PNG')
      response['overlay_png'] = base64.b64encode(buf.getvalue()).decode()
    return response

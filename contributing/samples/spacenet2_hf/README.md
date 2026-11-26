# SpaceNet 2 building footprints on Hugging Face

An ADK agent that works on [SpaceNet 2](https://spacenet.ai/spacenet-buildings-dataset-v2/)
(building footprint extraction from 30 cm WorldView-3 imagery of Las Vegas,
Paris, Shanghai and Khartoum) using only Hugging Face for compute:

| part | runs on |
|---|---|
| Building detector: Mask R-CNN R50-FPN from [`benjbritton/spacenet2-building-detector`](https://huggingface.co/benjbritton/spacenet2-building-detector) | a dedicated **HF Inference Endpoint** with the custom `endpoint/handler.py` |
| Agent LLM (default `openai/gpt-oss-120b`) | **HF Inference Providers**, through the OpenAI-compatible router at `router.huggingface.co/v1` |
| Imagery and labels | the public SpaceNet bucket on AWS (`s3://spacenet-dataset`, anonymous HTTPS) |

## Layout

```
spacenet2_hf/
├── agent.py              # root_agent and its five tools
├── spacenet2.py          # tiles, labels, endpoint client, SpaceNet F1
├── evaluate.py           # batch scoring / submission CSV, no LLM
└── endpoint/
    ├── handler.py        # Inference Endpoints custom handler
    ├── requirements.txt  # extra packages for the endpoint container
    ├── deploy_endpoint.py
    └── serve_local.py    # the same handler behind a local HTTP server
```

## The endpoint handler

The model is a Detectron2 checkpoint. Detectron2 has no wheels, so the handler
rebuilds the same network on plain torchvision and maps the weights onto it.
The endpoint image then needs only `pip install`. The port reproduces the
Detectron2 details that change the outputs:

- MSRA-style ResNet, with the stride on the 1x1 conv
- `ROIAlignV2` (`aligned=True`) and Detectron2's FPN level assignment
- unrounded anchors
- background-last class order
- BGR input with Detectron2's pixel mean

The checkpoint is loaded with `torch.load(weights_only=True)`.

16-bit PS-RGB GeoTIFFs go in as they are: the handler applies the training
stretch (2nd to 98th percentile per band) and returns the tile's affine
transform. Footprints come back as polygons in pixel coordinates that follow
the mask's pixel edges.

Request and response:

```json
POST {"inputs": "<base64 tif/png/jpg>",
      "parameters": {"score_threshold": 0.544, "return_overlay": false,
                     "max_detections": 100}}

{"num_buildings": 18, "image_size": [650, 650],
 "buildings": [{"score": 0.9987, "bbox": [...], "area_px": 4980,
                "polygon": [[x, y], ...]}],
 "georef": {"crs": "EPSG:4326", "transform": [a, b, c, d, e, f]}}
```

## Run it

```bash
pip install shapely                     # on top of ADK and litellm
export HF_TOKEN=hf_...

# 1a. Dedicated endpoint: needs a write token and billing on the namespace.
python endpoint/deploy_endpoint.py --dry-run
python endpoint/deploy_endpoint.py      # prints SPACENET2_ENDPOINT_URL
export SPACENET2_ENDPOINT_URL=https://<id>.<region>.aws.endpoints.huggingface.cloud

# 1b. Or the same handler locally (CPU is ~2 s per tile).
pip install torch torchvision opencv-python-headless shapely rasterio \
    fastapi uvicorn huggingface_hub
python endpoint/serve_local.py --port 8080   # SPACENET2_ENDPOINT_URL defaults here

# 2. Agent (from contributing/samples).
adk web .                               # pick spacenet2_hf
# optional: export SPACENET2_AGENT_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507

# 3. Batch scoring without the LLM.
python spacenet2_hf/evaluate.py --per-city 100 --out results.json
python spacenet2_hf/evaluate.py --split test --per-city 0 --submission test.csv
```

## Results

All numbers are on the model's full validation split, which it never trained
on: 2,118 tiles, of which 2,117 were scored (one request hit a gateway 502
before the client retried them). They were served by the dedicated endpoint
(`nvidia-t4`, about 7 minutes for the split) and scored with the SpaceNet F1
(polygon IoU ≥ 0.5) at a score threshold of 0.544. The model card's figures
come from Detectron2 on the same split.

| city | precision | recall | F1 (this endpoint) | F1 (model card) |
|---|---|---|---|---|
| Las Vegas | 0.929 | 0.858 | 0.892 | 0.894 |
| Paris | 0.821 | 0.722 | 0.769 | 0.776 |
| Shanghai | 0.739 | 0.627 | 0.678 | 0.683 |
| Khartoum | 0.679 | 0.576 | 0.623 | 0.626 |
| **macro** | | | **0.741** | **0.745** |

The torchvision port therefore reproduces the Detectron2 model to within
about 0.004 macro F1.

**Per-city score thresholds don't help.** Thresholds tuned per city on 200
training tiles each (Vegas 0.62, Paris 0.44, Shanghai 0.44, Khartoum 0.52)
give a validation macro F1 of 0.7404, against 0.7407 for the single 0.544
threshold. Even thresholds picked on the validation split itself would only
reach about 0.742. Further gains need a stronger detector, not
post-processing.

## Tools

| tool | what it does |
|---|---|
| `list_spacenet2_tiles` | Tile ids for `val`, `train` or `test`. Uses the model's own 80/20 split, so `val` tiles are unseen. |
| `detect_buildings` | Runs one tile. Saves an overlay PNG and a lon/lat GeoJSON as artifacts, and scores the tile if it has labels. |
| `evaluate_spacenet2` | SpaceNet F1: IoU ≥ 0.5, greedy matching by score, counts pooled per city, macro over cities. |
| `tune_score_threshold` | One endpoint pass per tile, then rescored at each candidate threshold. |
| `export_submission` | Writes a SpaceNet CSV (`ImageId,BuildingId,PolygonWKT_Pix,Confidence`) as an artifact. |

## Licences

The model weights and the SpaceNet data are CC BY-SA 4.0. `deploy_endpoint.py`
copies the weights into your own private repo with that licence and
attribution.

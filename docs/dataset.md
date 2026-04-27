# Dataset Preparation

The repository expects a `dataset/` directory at the repo root containing roadside camera videos and per-camera lane annotations. The training set used in the paper is **16 511 Wisconsin cameras / 132 lanes / 38 lane groups / 104,415 trajectories**, but the same pipeline works on any camera that follows the layout below.

## Expected layout

```
dataset/
├── 511video/                          # raw camera videos (one per camera)
│   ├── I43_Keefe.mp4
│   ├── I43_Walnut.mp4
│   ├── US12_Monona.mp4
│   └── …
│
├── camera_location_list.txt           # newline-separated camera names
│                                      # (basenames matching .mp4 files above)
│
└── preprocess/                        # per-camera hand-drawn annotations
    ├── I43_Keefe/
    │   └── annotation.json
    ├── I43_Walnut/
    │   └── annotation.json
    └── …
```

`dataset/` may be a symlink to a sibling directory (the development setup uses `dataset -> ../dataset`); the Makefile defaults follow this layout.  Override locations via the Makefile vars `VIDEO_DIR`, `PREPROCESS_DIR`, `CAMERA_LIST` if your data lives elsewhere.

## Acquiring the videos

The paper trains on archived 511 Wisconsin traffic-camera feeds (a public Wisconsin DOT service). Specific provenance and any sharing/redistribution constraints depend on what is permitted under the WisDOT terms in effect at download time — confirm those before sharing the dataset alongside the repo. Contact the authors (`tamaru@wisc.edu`) if you need the exact 16-camera collection that produced the paper's numbers.

The pipeline is camera-agnostic: any reasonably stationary roadside view at ≥10 fps with visible lane markings will work, given hand-drawn annotations in the schema below.

## `camera_location_list.txt`

Plain text, one camera identifier per line. The identifier is used as both the basename of `dataset/511video/{id}.mp4` and the per-camera directory name under `dataset/preprocess/{id}/`. Example:

```
I43_Keefe
I43_Walnut
US12_Monona
US12_Mineral
US12_Park
…
```

## `annotation.json` schema

`annotation.json` files are produced by the
**[Lanelet-Annotator](https://github.com/raynbowy23/Lanelet-Annotator)** GUI — the companion tool used to draw lane geometries against the camera's reference frame, group lanes by direction of travel, and record inter-lane relationships (e.g., merges). The annotator writes the JSON schema this loader expects, so the typical workflow is:

1. Run `make extract-cam CAMERA={cam}` to produce `last_frame.npy` for the new camera.
2. Open Lanelet-Annotator pointed at that camera's directory and draw the lanes / set per-group headings / mark merge or diverge relationships.
3. Save → the tool emits `dataset/preprocess/{cam}/annotation.json` in the schema below.

A `make annotate` target launches the tool from the sibling checkout it expects at `../graph_geolane_annotator/` (override with `ANNOT_DIR=…` on the make line, or invoke the annotator's own `main.py` directly if your clone lives elsewhere).

The full schema:

```jsonc
{
  "camera": "US12_Monona",                    // string, must match dir name
  "image": "last_frame.npy",                   // reference frame the
                                               // annotation is drawn against
                                               // (informational; the loader
                                               // does not require it)
  "lane_groups": [
    {
      "group_id": 0,                           // int, unique within camera
      "heading_deg": 87.5,                     // travel direction of this group
                                               // in image-frame degrees
      "lanes": [
        {
          "cls_id": 0,                         // int, lane index inside the group
          "color": [0, 255, 0],                // RGB triplet, used by viz only
          "waypoints": [
            {"x": 204.0, "y": 133.4},          // pixel coords on the reference
            {"x": 220.5, "y": 158.2},          // frame; ordered along travel
            …
          ]
        },
        …                                       // one entry per lane in the group
      ],
      "relationships": [                        // optional; used by relational
                                                // diffusion training
        {"from_lane": 0, "to_lane": 1,
         "type": "merge", "merge_point": 0.6}
      ]
    },
    …                                           // one entry per lane group
  ]
}
```

Required keys at load time (`src/data/annotation_loader.py`):

- top-level: `camera`, `lane_groups`
- per group: `group_id`, `lanes`
- per lane: `cls_id`, `waypoints`

`heading_deg` is required by the geometric assignment step (`scripts/run_assignment.py`); `relationships` is required only by the relational diffusion training (`make train-generation-relational-scratch`).  `color` and `image` are optional informational fields.

Pixel coordinates are in the same frame as `last_frame.npy`, which `extract_video.py` writes during preprocessing. They are normalized to `[0, 1]` internally by the data loader, so the annotation's source resolution does not have to match a fixed canvas size — annotate against the actual camera resolution.

## End-to-end preparation

Once `dataset/` is populated as above:

```bash
make extract       # video → trajectory.csv (per camera)
make assign        # tracklet → lane assignment (per camera)
```

`make extract` invokes `scripts/extract_video.py`, which runs YOLOv11n with persistent tracking and writes the following per camera under `--output` (default `results/preprocess/{camera}/`):

| File | Purpose |
|---|---|
| `trajectory.csv` | Main trajectory data. Columns: `id, time, frame_num, class, conf, x, y, w, h`. One row per detection. |
| `last_frame.npy` | Last processed frame; used as the reference for annotation pixel coordinates and as background for visualization figures. |
| `collect_cars.npy` | Detection bounding boxes (positions + frame ids). |
| `collect_det_dots_including_truck.npy` | Same, including class. |
| `trajectory_viz.png` | Sanity-check visualization with track polylines drawn on the first frame. Skip with `--no-viz`. |

YOLO weights (`yolo11n.pt`) are auto-downloaded by `ultralytics` on the first call; the file lands at the repo root and is gitignored.

`make assign` invokes `scripts/run_assignment.py`, which projects every tracklet onto each annotated lane in its group and assigns it to the nearest one (mean point-to-polyline distance under a 60-pixel threshold; rejected tracks are flagged out-of-bounds). Outputs land under `results/preprocessing/lane_assignment/`:

| File | Purpose |
|---|---|
| `{camera}/lane_assignments.csv` | Per-tracklet lane assignment with confidence. |
| `summary.csv` | Per-camera assignment yield and rejection rate. |

After both steps complete, `make train CONFIG=configs/lane_contrastive.yaml` will pick up the data and begin Stage-1 contrastive training.

## Adding a new camera

1. Drop the video at `dataset/511video/{new_camera}.mp4`.
2. Add `{new_camera}` as a new line in `dataset/camera_location_list.txt`.
3. Hand-annotate the lanes against the camera's reference frame and save to `dataset/preprocess/{new_camera}/annotation.json` following the schema above. The reference frame can be obtained by running `make extract-cam CAMERA={new_camera}` first — it produces `last_frame.npy` you can annotate against.
4. Re-run `make extract-cam CAMERA={new_camera}` (if not already done) followed by `make assign-cam CAMERA={new_camera}`.
5. Re-train, or add the new camera to the held-out set for a leave-one-camera-out evaluation pass.

## Troubleshooting

- **`No trajectory.csv at …`** — `make extract` wrote to a different directory than the encoder's data loader expects. Either pass the matching `--output` to `extract_video.py` or set `PREPROCESS_DIR` in the Makefile so both sides agree.
- **`No annotation for {camera}`** — missing or misnamed `dataset/preprocess/{camera}/annotation.json`. The loader logs the exact path it tried.
- **`Missing keys: {…}`** raised by `load_annotation_json` — schema validation failure. Compare against the required keys listed above (top-level `camera` + `lane_groups`; per-lane `cls_id` + `waypoints`).
- **Lane group never matched to annotation** — geometric assignment matches groups by `heading_deg`. Check that the annotation's heading matches the actual travel direction in the camera frame to within ~30°; the loader handles 180° flips automatically.

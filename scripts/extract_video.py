#!/usr/bin/env python3
"""
Standalone Video Extractor for GeoORBIT Lane Detection

This script extracts trajectory data from traffic camera videos and saves
.npy files that can be used for training. It can be run asynchronously
for multiple videos in parallel.

Usage:
    python scripts/extract_video.py --video path/to/video.mp4 --camera camera_loc
    python scripts/extract_video.py --video-dir dataset/511video --camera camera_loc
    python scripts/extract_video.py --list cameras.txt  # Process multiple cameras

Output Files:
    - {output_dir}/{camera}/trajectory.csv      - Vehicle trajectories
    - {output_dir}/{camera}/last_frame.npy      - Last processed frame
    - {output_dir}/{camera}/collect_cars.npy    - Vehicle detections
    - {output_dir}/{camera}/collect_det_dots_including_truck.npy - Extended detections

Example:
    # Extract single video
    python scripts/extract_video.py \\
        --video dataset/511video/camera_001.mp4 \\
        --camera camera_001 \\
        --output results/preprocess

    # Extract all cameras listed in file
    python scripts/extract_video.py \\
        --list dataset/camera_location_list.txt \\
        --video-dir dataset/511video \\
        --output results/preprocess \\
        --parallel 4
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, List, Tuple

import cv2
import numpy as np
import polars as pl

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_video(
    video_path: Path,
    camera_loc: str,
    output_dir: Path,
    detection_period: int = 3600,
    conf_threshold: float = 0.25,
    model_path: str = "yolo11n.pt",
    save_visualization: bool = True
) -> dict:
    """
    Extract trajectory data from a single video file.

    Args:
        video_path: Path to video file
        camera_loc: Camera location identifier
        output_dir: Output directory for .npy files
        detection_period: Seconds of video to process
        conf_threshold: YOLO confidence threshold
        model_path: Path to YOLO model
        save_visualization: Whether to save trajectory visualization

    Returns:
        Dict with extraction statistics
    """
    logger.info(f"[{camera_loc}] Starting extraction from {video_path}")

    # Setup output directory
    out_path = Path(output_dir, camera_loc)
    out_path.mkdir(parents=True, exist_ok=True)

    # Open video
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    logger.info(f"[{camera_loc}] Video: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")

    # Limit to detection_period
    max_frames = int(min(detection_period * fps, frame_count)) if fps > 0 else frame_count

    # Initialize YOLO tracker
    track_model = YOLO(model_path)
    track_history = defaultdict(lambda: [])

    # Collection lists
    collect_cars = []
    collect_det_dots_including_truck = []
    trajectory_data = []

    first_frame = None
    last_frame = None
    frame_id = 0

    logger.info(f"[{camera_loc}] Processing {max_frames} frames...")

    while frame_id < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        if first_frame is None:
            first_frame = frame.copy()
        last_frame = frame.copy()

        # Run YOLO tracking
        results = track_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=conf_threshold,
            classes=[2, 5, 7]  # car, bus, truck
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                x, y, w, h = box
                center_x, center_y = int(x), int(y)

                # Store trajectory point
                trajectory_data.append({
                    'id': track_id,
                    'time': frame_id / fps if fps > 0 else frame_id,
                    'frame_num': frame_id,
                    'class': cls,
                    'conf': conf,
                    'x': center_x,
                    'y': center_y,
                    'w': int(w),
                    'h': int(h)
                })

                # Store for continuous learning
                collect_cars.append((center_x, center_y, int(w), int(h), frame_id, conf))
                collect_det_dots_including_truck.append((center_x, center_y, int(w), int(h), frame_id, conf, cls))

                # Track history for visualization
                track = track_history[track_id]
                track.append((center_x, center_y))
                if len(track) > 30:
                    track.pop(0)

        frame_id += 1

        # Progress logging
        if frame_id % 500 == 0:
            logger.info(f"[{camera_loc}] Processed {frame_id}/{max_frames} frames ({100*frame_id/max_frames:.1f}%)")

    video.release()

    # Create trajectory DataFrame
    out_df = pl.DataFrame(trajectory_data)

    # Save outputs
    logger.info(f"[{camera_loc}] Saving outputs to {out_path}")

    # Save .npy files
    np.save(out_path / "last_frame.npy", last_frame)
    np.save(out_path / "collect_cars.npy", np.array(collect_cars, dtype=object))
    np.save(out_path / "collect_det_dots_including_truck.npy",
            np.array(collect_det_dots_including_truck, dtype=object))

    # Save trajectory CSV
    out_df.write_csv(out_path / "trajectory.csv")

    # Save visualization
    if save_visualization and first_frame is not None:
        viz_frame = first_frame.copy()
        for track_id, track in track_history.items():
            if len(track) > 1:
                points = np.array(track, dtype=np.int32)
                cv2.polylines(viz_frame, [points], False, (0, 255, 0), 2)
        cv2.imwrite(str(out_path / "trajectory_viz.png"), viz_frame)

    stats = {
        'camera': camera_loc,
        'frames_processed': frame_id,
        'trajectories': len(trajectory_data),
        'unique_vehicles': len(set(d['id'] for d in trajectory_data)) if trajectory_data else 0,
        'output_path': str(out_path)
    }

    logger.info(f"[{camera_loc}] Extraction complete: {stats['trajectories']} points, "
                f"{stats['unique_vehicles']} vehicles")

    return stats


def process_camera_wrapper(args: Tuple) -> dict:
    """Wrapper for parallel processing."""
    video_path, camera_loc, output_dir, detection_period, conf_threshold = args
    try:
        return extract_video(
            video_path=Path(video_path),
            camera_loc=camera_loc,
            output_dir=Path(output_dir),
            detection_period=detection_period,
            conf_threshold=conf_threshold
        )
    except Exception as e:
        logger.error(f"[{camera_loc}] Extraction failed: {e}")
        return {'camera': camera_loc, 'error': str(e)}


def find_video_for_camera(video_dir: Path, camera_loc: str) -> Optional[Path]:
    """Find video file for a camera location."""
    # Try common patterns
    patterns = [
        f"{camera_loc}.mp4",
        f"{camera_loc}.avi",
        f"{camera_loc}.mov",
        f"{camera_loc}/*.mp4",
        f"{camera_loc}/*.avi",
    ]

    for pattern in patterns:
        matches = list(video_dir.glob(pattern))
        if matches:
            return matches[0]

    # Try finding directory with videos
    cam_dir = video_dir / camera_loc
    if cam_dir.is_dir():
        for ext in ['.mp4', '.avi', '.mov']:
            videos = list(cam_dir.glob(f"*{ext}"))
            if videos:
                return videos[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectory data from traffic camera videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str,
                             help='Path to single video file')
    input_group.add_argument('--list', type=str,
                             help='Path to file with camera locations (one per line)')

    # Additional options
    parser.add_argument('--camera', type=str,
                        help='Camera location ID (required with --video)')
    parser.add_argument('--video-dir', type=str, default='./dataset/511video',
                        help='Directory containing videos (used with --list)')
    parser.add_argument('--output', '-o', type=str, default='./results/preprocess',
                        help='Output directory for .npy files')
    parser.add_argument('--detection-period', type=int, default=3600,
                        help='Seconds of video to process (default: 3600)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--parallel', '-j', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip saving trajectory visualization')

    args = parser.parse_args()

    # Validate arguments
    if args.video and not args.camera:
        parser.error("--camera is required when using --video")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single video mode
    if args.video:
        stats = extract_video(
            video_path=Path(args.video),
            camera_loc=args.camera,
            output_dir=output_dir,
            detection_period=args.detection_period,
            conf_threshold=args.conf_threshold,
            save_visualization=not args.no_viz
        )
        print(f"\nExtraction complete: {stats}")
        return

    # Batch mode from camera list
    camera_list_path = Path(args.list)
    if not camera_list_path.exists():
        logger.error(f"Camera list not found: {camera_list_path}")
        sys.exit(1)

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        sys.exit(1)

    # Read camera list
    with open(camera_list_path) as f:
        cameras = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(cameras)} cameras from {camera_list_path}")

    # Build task list
    tasks = []
    for camera in cameras:
        video_path = find_video_for_camera(video_dir, camera)
        if video_path:
            tasks.append((
                str(video_path),
                camera,
                str(output_dir),
                args.detection_period,
                args.conf_threshold
            ))
        else:
            logger.warning(f"No video found for camera: {camera}")

    if not tasks:
        logger.error("No videos found to process")
        sys.exit(1)

    # Process videos
    results = []
    if args.parallel > 1:
        logger.info(f"Processing {len(tasks)} videos with {args.parallel} workers")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_camera_wrapper, task): task[1]
                       for task in tasks}
            for future in as_completed(futures):
                camera = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[{camera}] Failed: {e}")
                    results.append({'camera': camera, 'error': str(e)})
    else:
        for task in tasks:
            result = process_camera_wrapper(task)
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    success = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"Successful: {len(success)}/{len(results)}")
    for r in success:
        print(f"  {r['camera']}: {r.get('trajectories', 0)} points, "
              f"{r.get('unique_vehicles', 0)} vehicles")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  {r['camera']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

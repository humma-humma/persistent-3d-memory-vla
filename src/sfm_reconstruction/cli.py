from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import (
    load_image_dataset,
    load_stage1_dataset,
    load_stage2_dataset,
)
from .io import (
    write_camera_parameters,
    write_point_cloud,
    write_rich_point_cloud,
    write_summary,
)
from .matching import MatchingConfig, generate_correspondences
from .reconstruction import ReconstructionConfig, ReconstructionResult, reconstruct
from .reprojection_diagnostics import write_reprojection_diagnostics
from .stage2_pose_refinement import (
    Stage2PoseRefinementConfig,
    refine_stage2_poses,
    write_stage2_pose_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incremental calibrated Structure-from-Motion."
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--no-bundle-adjustment",
        action="store_true",
        help="Skip final global bundle adjustment.",
    )
    parser.add_argument("--bundle-adjustment-max-nfev", type=int, default=40)
    parser.add_argument("--min-pnp-points", type=int, default=12)
    parser.add_argument("--min-pnp-inliers", type=int, default=10)
    parser.add_argument("--min-track-observations", type=int, default=2)
    parser.add_argument("--max-reprojection-error", type=float, default=5.0)
    parser.add_argument("--max-point-distance-factor", type=float)
    parser.add_argument("--two-view-max-reprojection-error", type=float)
    parser.add_argument("--two-view-min-triangulation-angle", type=float)
    parser.add_argument("--matching-cache", type=Path)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument(
        "--feature-mode",
        choices=("sift", "akaze", "sift+akaze", "superpoint-lightglue"),
        default="sift",
    )
    parser.add_argument(
        "--learned-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--learned-filter-threshold", type=float, default=0.2)
    parser.add_argument("--learned-cycle-filter", action="store_true")
    parser.add_argument("--learned-min-cycle-matches", type=int, default=15)
    parser.add_argument("--learned-augment-supplied", action="store_true")
    parser.add_argument("--initial-pose-correspondence-dir", type=Path)
    parser.add_argument("--wide-baseline", action="store_true")
    parser.add_argument("--wide-pose-only", action="store_true")
    parser.add_argument("--wide-retrieval-max-pairs", type=int, default=100)
    parser.add_argument("--wide-min-frame-gap", type=int, default=4)
    parser.add_argument("--wide-min-similarity", type=float, default=0.7)
    parser.add_argument("--wide-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--wide-min-spatial-coverage", type=float, default=0.25)
    parser.add_argument("--wide-min-cycle-matches", type=int, default=15)
    parser.add_argument("--wide-max-pairs-per-image", type=int, default=1)
    parser.add_argument("--wide-pose-rotation-weight", type=float, default=0.15)
    parser.add_argument("--wide-pose-translation-weight", type=float, default=0.05)
    parser.add_argument(
        "--wide-retrieval-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--wide-retrieval-batch-size", type=int, default=16)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.03)
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    parser.add_argument("--pair-window", type=int, default=3)
    parser.add_argument(
        "--matching-pair-source",
        choices=("circular", "supplied"),
        default="circular",
    )
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--matching-essential-threshold", type=float, default=1.0)
    parser.add_argument("--min-match-inliers", type=int, default=15)
    parser.add_argument("--mask-apriltags", action="store_true")
    parser.add_argument("--apriltag-padding", type=float, default=0.12)
    parser.add_argument("--overwrite-correspondences", action="store_true")
    parser.add_argument("--write-reprojection-diagnostics", action="store_true")
    parser.add_argument("--reprojection-overlay-limit", type=int, default=300)
    parser.add_argument(
        "--write-incremental-memory",
        action="store_true",
        help=(
            "Export each same-map registration snapshot and convert the "
            "sequence into persistent-memory and adapter JSON."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 1:
        dataset = load_stage1_dataset(args.dataset).subset(args.max_images)
    else:
        image_dataset = load_image_dataset(args.dataset).subset(args.max_images)
        cache_root = args.matching_cache or args.output_dir / "matching_cache"
        correspondence_dir, matching_summary = generate_correspondences(
            image_dataset,
            cache_root,
            MatchingConfig(
                max_features=args.max_features,
                feature_mode=args.feature_mode,
                sift_contrast_threshold=args.sift_contrast_threshold,
                sift_edge_threshold=args.sift_edge_threshold,
                pair_window=args.pair_window,
                pair_source=args.matching_pair_source,
                ratio_threshold=args.ratio_threshold,
                essential_threshold=args.matching_essential_threshold,
                min_inliers=args.min_match_inliers,
                mask_apriltags=args.mask_apriltags,
                apriltag_padding=args.apriltag_padding,
                learned_device=args.learned_device,
                learned_filter_threshold=args.learned_filter_threshold,
                learned_cycle_filter=args.learned_cycle_filter,
                learned_min_cycle_matches=args.learned_min_cycle_matches,
                learned_augment_supplied=args.learned_augment_supplied,
                wide_baseline=args.wide_baseline,
                wide_pose_only=args.wide_pose_only,
                wide_retrieval_max_pairs=args.wide_retrieval_max_pairs,
                wide_min_frame_gap=args.wide_min_frame_gap,
                wide_min_similarity=args.wide_min_similarity,
                wide_min_inlier_ratio=args.wide_min_inlier_ratio,
                wide_min_spatial_coverage=args.wide_min_spatial_coverage,
                wide_min_cycle_matches=args.wide_min_cycle_matches,
                wide_max_pairs_per_image=args.wide_max_pairs_per_image,
                wide_retrieval_device=args.wide_retrieval_device,
                wide_retrieval_batch_size=args.wide_retrieval_batch_size,
            ),
            overwrite=args.overwrite_correspondences,
        )
        print(
            f"Generated {matching_summary.correspondences} correspondences "
            f"across {matching_summary.accepted_pairs} image pairs."
        )
        dataset = load_stage2_dataset(
            args.dataset, correspondence_dir
        ).subset(args.max_images)

    config = ReconstructionConfig(
        bundle_adjustment=not args.no_bundle_adjustment,
        bundle_adjustment_max_nfev=args.bundle_adjustment_max_nfev,
        min_pnp_points=args.min_pnp_points,
        min_pnp_inliers=args.min_pnp_inliers,
        min_track_observations=args.min_track_observations,
        max_reprojection_error=args.max_reprojection_error,
        max_point_distance_factor=args.max_point_distance_factor,
        two_view_max_reprojection_error=args.two_view_max_reprojection_error,
        two_view_min_triangulation_angle=args.two_view_min_triangulation_angle,
        initial_pose_correspondence_dir=args.initial_pose_correspondence_dir,
    )
    snapshot_paths: list[Path] = []
    snapshot_manifest: list[dict] = []

    def write_snapshot(registered_image_ids, snapshot_result) -> None:
        step = len(snapshot_paths)
        snapshot_dir = args.output_dir / "incremental_snapshots" / f"frame_{step:06d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        rich_path = snapshot_dir / "estimated_points_rich.ply"
        observed_path = snapshot_dir / "observed_points_rich.ply"
        write_camera_parameters(
            snapshot_dir / "estimated_camera_parameters.json",
            dataset,
            snapshot_result,
        )
        write_rich_point_cloud(rich_path, dataset, snapshot_result)
        observed_track_ids = {
            track_id
            for track_id in snapshot_result.points
            if set(registered_image_ids)
            & set(snapshot_result.tracks[track_id].observations)
        }
        observed_result = ReconstructionResult(
            poses=snapshot_result.poses,
            points={
                track_id: snapshot_result.points[track_id]
                for track_id in observed_track_ids
            },
            tracks=snapshot_result.tracks,
            initial_pair=snapshot_result.initial_pair,
            skipped_track_conflicts=snapshot_result.skipped_track_conflicts,
        )
        write_rich_point_cloud(observed_path, dataset, observed_result)
        snapshot_paths.append(observed_path)
        snapshot_manifest.append(
            {
                "frame_index": step,
                "registered_image_ids": list(registered_image_ids),
                "registered_cameras": len(snapshot_result.poses),
                "points": len(snapshot_result.points),
                "observed_points": len(observed_result.points),
                "rich_point_cloud": str(rich_path.relative_to(args.output_dir)),
                "memory_observations": str(observed_path.relative_to(args.output_dir)),
            }
        )

    result = reconstruct(
        dataset,
        config,
        snapshot_callback=write_snapshot if args.write_incremental_memory else None,
    )
    if args.write_incremental_memory:
        snapshot_root = args.output_dir / "incremental_snapshots"
        (snapshot_root / "manifest.json").write_text(
            json.dumps(
                {
                    "frame_semantics": (
                        "Registration events before final bundle adjustment; "
                        "frame 0 registers the initialization pair."
                    ),
                    "snapshots": snapshot_manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        from persistent_scene_memory.reconstruction_bridge import (
            process_reconstruction_sequence,
        )

        process_reconstruction_sequence(
            snapshot_paths,
            args.output_dir / "incremental_memory",
        )
    if args.stage == 2 and args.wide_pose_only:
        refinement_config = Stage2PoseRefinementConfig(
            wide_rotation_weight=args.wide_pose_rotation_weight,
            wide_translation_weight=args.wide_pose_translation_weight,
        )
        result, refinement_summary, refinement_diagnostics = refine_stage2_poses(
            dataset,
            result,
            cache_root / "wide_correspondences",
            config,
            refinement_config,
        )
        write_stage2_pose_refinement(
            args.output_dir,
            refinement_config,
            refinement_summary,
            refinement_diagnostics,
        )
        print(
            f"Accepted {refinement_summary.accepted_constraints}/"
            f"{refinement_summary.candidate_constraints} wide pose constraints."
        )

    write_camera_parameters(
        args.output_dir / "estimated_camera_parameters.json", dataset, result
    )
    write_point_cloud(args.output_dir / "estimated_points.ply", result)
    write_rich_point_cloud(
        args.output_dir / "estimated_points_rich.ply", dataset, result
    )
    write_summary(args.output_dir / "summary.json", dataset, result)
    if args.write_reprojection_diagnostics:
        write_reprojection_diagnostics(
            args.output_dir / "reprojection_diagnostics",
            dataset,
            result,
            max_overlay_points=args.reprojection_overlay_limit,
        )

    print(
        f"Registered {len(result.poses)}/{len(dataset.image_paths)} cameras; "
        f"reconstructed {len(result.points)} points."
    )
    print(f"Outputs: {args.output_dir.resolve()}")

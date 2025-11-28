import argparse
from pathlib import Path


def count_vertices_from_ply(ply_path: Path) -> int:
    with ply_path.open("rb") as fp:
        for raw_line in fp:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) >= 3 and parts[2].isdigit():
                    return int(parts[2])
            if line == "end_header":
                break
    raise ValueError(f"Could not determine vertex count from {ply_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count gaussians in a PLY checkpoint.")
    parser.add_argument("ply_path", type=Path, help="Path to the point_cloud.ply file.")
    args = parser.parse_args()

    ply_path = args.ply_path
    if not ply_path.exists():
        raise SystemExit(f"PLY file not found: {ply_path}")

    count = count_vertices_from_ply(ply_path)
    print(f"Number of Gaussians: {count}")


if __name__ == "__main__":
    main()

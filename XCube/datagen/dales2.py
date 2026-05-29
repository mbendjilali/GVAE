import os
import glob
import numpy as np
import torch
import laspy
import open3d as o3d
import fvdb
from fvdb import JaggedTensor
from tqdm import tqdm
from xcube.utils.color_util import semantic_from_points


def read_laz(path):
    """
    Returns:
        xyz       : np.ndarray  [N, 3]  float64  – world coordinates
        labels    : np.ndarray  [N]     int32    – 0-based semantic class (0..14)
        instances : np.ndarray  [N]     int32    – instance ID (0 = no instance)
    """
    las = laspy.read(path)

    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)

    labels = np.array(las.classification, dtype=np.int32)
    instances = np.array(las.instance, dtype=np.int32)

    return xyz, labels, instances

def estimate_normals(xyz, search_radius=1.0):
    """
    xyz : np.ndarray [N, 3]
    Returns normals : np.ndarray [N, 3]  unit vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=search_radius)
    )

    # Orient all normals toward the sky (consistent direction)
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 1e6])
    )

    normals = np.asarray(pcd.normals).astype(np.float32)
    return normals

def generate_crops(xyz, labels, instances, normals, crop_size=100.0, stride=80.0):
    """
    Slide a crop_size × crop_size window over the XY plane with a given stride.
    Yields (xyz_crop, labels_crop, instances_crop, normals_crop) for each window
    that has at least 1000 points.
    """
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    x_start = x_min
    while x_start + crop_size <= x_max + stride:
        y_start = y_min
        while y_start + crop_size <= y_max + stride:
            mask = (
                (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_start + crop_size) &
                (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_start + crop_size)
            )
            if mask.sum() >= 1000:   # skip nearly-empty crops
                yield (
                    xyz[mask],
                    labels[mask],
                    instances[mask],
                    normals[mask],
                )
            y_start += stride
        x_start += stride


def voxelize_and_save(xyz, normals, labels, instances, voxel_size, out_path, build_splatting):
    """
    xyz         : np.ndarray [N, 3]
    normals     : np.ndarray [N, 3]
    labels      : np.ndarray [N]   int   0-based
    instances   : np.ndarray [N]   int
    voxel_size  : float  (e.g. 0.4 for coarse, 0.1 for fine)
    out_path    : str    where to write the .pkl file
    build_splatting : bool   True for coarse, False for fine
    """

    # Centre the crop around zero so voxel indices start near the origin
    origin_offset = xyz.min(axis=0)
    xyz_centred = xyz - origin_offset

    # Convert to torch float32 tensors and move to GPU
    input_xyz      = torch.from_numpy(xyz_centred).float().cuda()
    input_normal   = torch.from_numpy(normals).float().cuda()
    input_semantic = torch.from_numpy(labels).long().cuda().unsqueeze(1)   # [N, 1]
    input_instance = torch.from_numpy(instances).long().cuda().unsqueeze(1) # [N, 1]

    # --- Build the voxel grid ---
    origins = [voxel_size / 2.0] * 3   # voxel centres are offset by half a voxel

    if build_splatting:
        # Coarse: each voxel gets the nearest point (merges duplicates)
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
            JaggedTensor(input_xyz),
            voxel_sizes=voxel_size,
            origins=origins
        )
    else:
        # Fine: one voxel per occupied cell
        target_grid = fvdb.sparse_grid_from_points(
            JaggedTensor(input_xyz),
            voxel_sizes=voxel_size,
            origins=origins
        )

    # --- Assign normals to voxels via trilinear splatting ---
    target_normal = target_grid.splat_trilinear(
        JaggedTensor(input_xyz),
        JaggedTensor(input_normal)
    )
    # Normalize so each voxel has a unit normal
    target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)

    # --- Assign semantics to voxels (nearest-neighbour lookup) ---
    target_xyz = target_grid.grid_to_world(target_grid.ijk.float()).jdata
    target_semantic  = semantic_from_points(target_xyz, input_xyz, input_semantic).long()
    target_instance  = semantic_from_points(target_xyz, input_xyz, input_instance).long()

    # --- Save ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_dict = {
        "points":    target_grid.to("cpu"),
        "normals":   target_normal.cpu(),
        "semantics": target_semantic[:, 0].cpu(),
        "instances": target_instance[:, 0].cpu(),
    }
    torch.save(save_dict, out_path)

    # Compute grid resolution for folder names
def get_grid_resolution(crop_size, voxel_size):
    return int(round(crop_size / voxel_size))


if __name__ == "__main__":
    INPUT_DIR  = "data/raw/dales2"   
    OUTPUT_DIR = "data/preprocessed"  #
    CROP_SIZE  = 100.0   # XY window size in metres
    STRIDE     = 80.0    # sliding window stride in metres
    VOXEL_SIZE_COARSE = 5 # 5 m
    VOXEL_SIZE_FINE   = 1 # 1 m


    coarse_grid_res = get_grid_resolution(CROP_SIZE, VOXEL_SIZE_COARSE)
    fine_grid_res   = get_grid_resolution(CROP_SIZE, VOXEL_SIZE_FINE)

    for split in ["train", "test"]:
        split_input_dir = os.path.join(INPUT_DIR, split)
        all_laz = sorted(glob.glob(os.path.join(split_input_dir, "*.laz")))
        print(f"\n[{split}] Found {len(all_laz)} .laz files")

        # Use dynamic folder names based on grid resolution
        coarse_dir = os.path.join(OUTPUT_DIR, f"{coarse_grid_res}", split)
        fine_dir   = os.path.join(OUTPUT_DIR, f"{fine_grid_res}", split)
        os.makedirs(coarse_dir, exist_ok=True)
        os.makedirs(fine_dir,   exist_ok=True)

        split_stems = []

        for laz_path in tqdm(all_laz, desc=f"Tiles [{split}]"):
            tile_name = os.path.splitext(os.path.basename(laz_path))[0]

            print(f"  Reading {tile_name} ...")
            xyz, labels, instances = read_laz(laz_path)

            print(f"  Estimating normals ({len(xyz):,} points) ...")
            normals = estimate_normals(xyz, search_radius=1.0)

            crop_idx = 0
            for xyz_c, labels_c, instances_c, normals_c in generate_crops(
                    xyz, labels, instances, normals,
                    crop_size=CROP_SIZE, stride=STRIDE):

                stem = f"{tile_name}_crop{crop_idx}"
                crop_idx += 1

                coarse_path = os.path.join(coarse_dir, stem + ".pkl")
                fine_path   = os.path.join(fine_dir,   stem + ".pkl")

                # Coarse: build_splatting=True
                voxelize_and_save(xyz_c, normals_c, labels_c, instances_c,
                                  voxel_size=VOXEL_SIZE_COARSE,
                                  out_path=coarse_path,
                                  build_splatting=True)

                # Fine: build_splatting=False
                voxelize_and_save(xyz_c, normals_c, labels_c, instances_c,
                                  voxel_size=VOXEL_SIZE_FINE,
                                  out_path=fine_path,
                                  build_splatting=False)

                split_stems.append(stem)

        lst_path = os.path.join(OUTPUT_DIR, f"{split}.lst")
        with open(lst_path, "w") as f:
            f.write("\n".join(split_stems))
        print(f"  Wrote {len(split_stems)} crops → {lst_path}")
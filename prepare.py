import os
import shutil

# Path to the folder that directly contains the class folders
# e.g. dataset/AMDNet23 if your structure is dataset/AMDNet23/Normal, Cataract...
AMDNET_ROOT = os.path.join("dataset", "AMDNet23")  # change if needed

OUT_ROOT = "binary_data"

def main():
    normal_src = os.path.join(AMDNET_ROOT, "Normal")
    cataract_src = os.path.join(AMDNET_ROOT, "Cataract")

    if not os.path.isdir(normal_src) or not os.path.isdir(cataract_src):
        raise FileNotFoundError(
            f"Could not find 'Normal' and 'Cataract' folders under {AMDNET_ROOT}.\n"
            f"Make sure AMDNET_ROOT points to the directory that contains these folders."
        )

    # Start fresh
    if os.path.exists(OUT_ROOT):
        print(f"Removing existing '{OUT_ROOT}'...")
        shutil.rmtree(OUT_ROOT)

    normal_out = os.path.join(OUT_ROOT, "normal")
    cataract_out = os.path.join(OUT_ROOT, "cataract")
    os.makedirs(normal_out, exist_ok=True)
    os.makedirs(cataract_out, exist_ok=True)

    def copy_images(src_dir, dst_dir):
        count = 0
        for fname in os.listdir(src_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(src_dir, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copy(src, dst)
                count += 1
        return count

    n_normal = copy_images(normal_src, normal_out)
    n_cataract = copy_images(cataract_src, cataract_out)

    print("✅ Built binary_data from AMDNet23")
    print("Normal images  :", n_normal)
    print("Cataract images:", n_cataract)
    print("binary_data folder:", os.path.abspath(OUT_ROOT))


if __name__ == "__main__":
    main()

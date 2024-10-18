import glob
import json
import os
import sys
from pathlib import Path


class MaterialsDB:
    def __init__(self, materials_dir):
        self.materials = {}
        if not materials_dir:  # empty database
            return
        mat_jsons = glob.glob(os.path.join(materials_dir, "*", "*.json"))
        for mat_json_file in mat_jsons:
            mat_name = Path(mat_json_file).parts[-2]
            mat_json = json.load(open(mat_json_file))
            mat_json["name"] = mat_name
            mat_json["diffuse"] = mat_json.get("color", "#ffffff")
            del mat_json["color"]
            albedo_path = os.path.join(materials_dir, mat_name, "albedo.png")
            if os.path.exists(albedo_path):
                mat_json["texture"] = f"ai2Texture.{mat_name}"
            self.materials[mat_name] = mat_json
            # print(mat_name, mat_json)

    def stk_mat(self, procthormat):
        canonical_name = procthormat["name"].replace(" ", "-")
        if canonical_name in self.materials:
            mat = self.materials[canonical_name]
        else:
            print(f'Warning material name "{canonical_name}" not found; using default.', file=sys.stderr)
            mat = {"name": canonical_name, "diffuse": "#ffffff"}

        def clamp(x):
            return int(min(max(0, x * 256), 255))

        if "color" in procthormat:  # override color
            c = procthormat["color"]
            rgb = (clamp(c["r"]), clamp(c["g"]), clamp(c["b"]))
            mat["diffuse"] = "#%02x%02x%02x" % rgb
        return mat

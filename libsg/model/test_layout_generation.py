"""
generate_scenes.py
---
One-off script to test scene generation outside of the context of libsg.
"""

import argparse
import json
import torch

from libsg.model.atiss import atiss_network, descale_bbox_params, square_room_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scenes")
    parser.add_argument("--config", help="Model configuration")
    parser.add_argument("--weight_file", default=None, help="Pretrained model weights")
    parser.add_argument("--output", help="Output scene")
    args = parser.parse_args()

    config = json.load(open(args.config))
    room_dim = int(config["data"]["room_layout_size"].split(",")[0])
    classes = config["data"]["classes"]  # TODO: save more cleanly along with bounds
    bounds = config["data"]["bounds"]
    config["weight_file"] = config.get("weight_file", args.weight_file)
    network = atiss_network(config=config)
    network.eval()

    room_mask = square_room_mask(room_dim=room_dim)
    bbox_params = network.generate_boxes(room_mask=room_mask)
    boxes = descale_bbox_params(bounds, bbox_params)
    bbox_params_t = (
        torch.cat([boxes["class_labels"], boxes["translations"], boxes["sizes"], boxes["angles"]], dim=-1)
        .cpu()
        .numpy()
    )
    objects = []
    for j in range(1, bbox_params_t.shape[1] - 1):
        objects.append(
            {
                "wnsynsetkey": classes[bbox_params_t[0, j, :-7].argmax(-1)],
                "dimensions": bbox_params_t[0, j, -4:-1].tolist(),
                "position": bbox_params_t[0, j, -7:-4].tolist(),
                "orientation": bbox_params_t[0, j, -1],
            }
        )
    json.dump(objects, open(args.output, "w"))

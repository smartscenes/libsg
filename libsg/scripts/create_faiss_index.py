import json
import os

import click
import faiss
import h5py
import numpy as np
import pandas as pd


def create_faiss_openshape(
    fp_embeddings_file, fp_ids_file, threedfuture_object_ids, four_embeddings_file, four_ids_file, output_dir
):
    with h5py.File(fp_embeddings_file, "r") as f:
        embeddings = f["pcd_features"][:]

    index = faiss.IndexFlatIP(embeddings.shape[-1])  # dim 1280
    index.add(embeddings)

    # load IDs for HSSD assets
    ids_df = pd.read_csv(fp_ids_file, names=["id"])
    ids_df["id"] = ids_df["id"].apply(lambda x: f"fpModel.{x}")
    ids_df["source"] = "fpModel"

    # load IDs for combined dataset (ABO, Objaverse, 3D-FUTURE, and ShapeNet)
    # and filter 3D-FUTURE IDs
    with open(threedfuture_object_ids, "r") as f:
        object_ids = json.load(f)

    four_ids_df = pd.read_csv(four_ids_file, names=["source", "unknown", "id"])
    threedf_mask = four_ids_df["source"] == "3D-FUTURE"
    num_with_embedding = threedf_mask.sum()
    threedf_mask = threedf_mask & (four_ids_df["id"].isin([model_id[9:] for model_id in object_ids]))
    num_valid = threedf_mask.sum()
    print(f"Number of valid 3D-FUTURE objects: {num_valid} ({num_with_embedding} original)")
    threedf_indices = four_ids_df[threedf_mask].index.tolist()
    threedf_ids_df = four_ids_df[threedf_mask][["id"]]
    threedf_ids_df["id"] = threedf_ids_df["id"].apply(lambda x: f"3dfModel.{x}")
    threedf_ids_df["source"] = "3dfModel"
    ids_df = pd.concat([ids_df, threedf_ids_df], ignore_index=True)

    # insert only the embeddings corresponding to 3D-FUTURE models
    with h5py.File(four_embeddings_file, "r") as f:
        four_embeddings = f["pcd_features"]
        index.add(four_embeddings[threedf_indices])

    print(f"Size of openshape index: {index.ntotal}")

    # write out final files
    embeddings_output = os.path.join(output_dir, "openshape_p_1280.index")
    faiss.write_index(index, embeddings_output)
    ids_output = os.path.join(output_dir, "openshape_p_1280.csv")
    ids_df.to_csv(ids_output, index=False)


def create_faiss_diffuscene(threedfuture_object_ids, threedfuture_object_dir, latent_dim, output_dir):
    with open(threedfuture_object_ids, "r") as f:
        object_ids = json.load(f)

    embeddings = []
    ids = []
    for model_id in object_ids:
        latent_file = os.path.join(threedfuture_object_dir, model_id[9:], f"raw_model_norm_pc_lat{latent_dim}.npz")
        embedding = np.load(latent_file)

        embeddings.append(embedding["latent"])
        ids.append(["3dfModel", model_id])

    # build embeddings
    embeddings = np.array(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[-1])
    index.add(embeddings)
    print(f"Size of diffuscene index: {index.ntotal}")

    # build ID lookup
    ids_df = pd.DataFrame(ids, columns=["source", "id"])

    # write out final files
    embeddings_output = os.path.join(output_dir, f"diffuscene_latent_p_{latent_dim}.index")
    faiss.write_index(index, embeddings_output)
    ids_output = os.path.join(output_dir, f"diffuscene_latent_p_{latent_dim}.csv")
    ids_df.to_csv(ids_output, index=False)


@click.command()
@click.option(
    "--openshape-fpmodel",
    "openshape_fp",
    default="../data/embeddings/fpmodels.h5",
    help="hdf5 file for OpenShape embeddings for HSSD assets",
)
@click.option(
    "--openshape-four",
    "openshape_four",
    default="../data/embeddings/four.h5",
    help="hdf5 file for OpenShape embeddings for aggregate dataset used to train OpenShape (Objaverse, ShapeNet, 3D-FUTURE, ABO)",
)
@click.option("-o", "--output-dir", "output_dir", default=".data/embeddings")
@click.option(
    "--fpmodel-ids",
    "fp_ids",
    default="../data/embeddings/fpmodels.csv",
    help="CSV file for list of IDs for HSSD assets",
)
@click.option(
    "--four-ids",
    "four_ids",
    default="../data/embeddings/four.csv",
    help="CSV file for list of IDs for aggregate dataset used to train OpenShape (Objaverse, ShapeNet, 3D-FUTURE, ABO)",
)
@click.option(
    "--threedfuture-object-ids",
    "threedfuture_object_ids",
    default="../data/diffuscene_stk/3d-future_object_ids.json",
    help="JSON file with list of valid 3D-FUTURE object IDs",
)
@click.option(
    "--threedfuture-object-dir",
    "threedfuture_object_dir",
    default="/project/3dlg-hcvc/semdifflayout/data/3D-FUTURE-model",
)
@click.option("--diffuscene-latent-dim", "latent_dim", default=32, help="latent dimension")
def main(
    openshape_fp,
    openshape_four,
    fp_ids,
    four_ids,
    threedfuture_object_ids,
    threedfuture_object_dir,
    latent_dim,
    output_dir,
):
    print("Creating Faiss indices for OpenShape...")
    create_faiss_openshape(openshape_fp, fp_ids, threedfuture_object_ids, openshape_four, four_ids, output_dir)
    print("Creating Faiss indices for DiffuScene...")
    create_faiss_diffuscene(threedfuture_object_ids, threedfuture_object_dir, latent_dim, output_dir)


if __name__ == "__main__":
    main()

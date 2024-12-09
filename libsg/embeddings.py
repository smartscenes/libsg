import faiss
import numpy as np
import pandas as pd


class EmbeddingsLookup:
    def __init__(self, ids_file: str, index_file: str, top_k: int = 10):
        self.ids = pd.read_csv(ids_file)
        self.index = faiss.read_index(index_file)
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.ids)

    def __call__(
        self, query: np.ndarray, max_retrieve: int = 0, sources: list[str] = None, model_ids: list[str] = None
    ) -> list[str]:
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        top_k = max(max_retrieve, self.top_k)
        if sources or model_ids:
            valid_ids = set()
            if sources:
                valid_ids = valid_ids.union(set(self.ids[self.ids["source"].isin(sources)].index.tolist()))
            else:
                valid_ids = set(self.ids.index.tolist())
            if model_ids:
                model_ids_by_index = set(self.ids[self.ids["id"].isin(model_ids)].index.tolist())
                valid_ids = valid_ids.intersection(model_ids_by_index)
            valid_ids = list(valid_ids)

            id_selector = faiss.IDSelectorArray(valid_ids)
            _, nearest_neighbors = self.index.search(
                query, self.top_k, params=faiss.SearchParametersIVF(sel=id_selector)
            )
        else:
            _, nearest_neighbors = self.index.search(query, top_k)

        return self.ids.iloc[nearest_neighbors[0, :]]["id"].tolist()

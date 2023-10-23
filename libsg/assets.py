from libsg.scene_types import JSONDict

import csv
import glob
import numpy as np
import os
import pysolr
from easydict import EasyDict


class AssetGroup:
    def __init__(self, id, metadata=None):
        self.id = id
        self.align_to = None
        self.scale_to = 1
        self.center_to = None
        if metadata:
            reader = csv.DictReader(open(metadata, 'r'))
            self._metadata = {}
            for row in reader:
                id = row['id']
                for k in row:
                    if k in ['maxX','maxY','maxZ','minX','minY','minZ','dimsX','dimsY','dimsZ']:
                        row[k] = float(row[k])
                self._metadata[id] = row

    def set_align_to(self, up, front):
        self.align_to = { 'up': up, 'front': front }

    def model_info(self, model_id):
        if '.' in model_id:
            tokens = model_id.split('.')
            source = tokens[0]
            id = tokens[1]
        else:
            source = self.id
            id = model_id
        model_info = {'source': source, 'id': id}
        if self._metadata and id in self._metadata:
            model_info = self._metadata[id]
        return model_info

    def has_transforms(self) -> bool:
        return self.align_to or (self.scale_to is not None and self.scale_to != 1) or self.center_to

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj[self.id] = {
            'alignTo': self.align_to,
            'scaleTo': self.scale_to,
            'centerTo': self.center_to
        }
        return obj


class AssetDb:
    def __init__(self, cfg, solr_url=None):
        path = cfg.get('path')
        self.assets = {}
        if path:
            # TODO: this is specific for scenestates
            scenestates = glob.glob(path + '/*.json')
            for s in scenestates:
                base = os.path.basename(s)
                uuid = base.split('.')[0]
                self.assets[uuid] = s
        self.defaults = cfg.get('defaults', {})
        self.config = cfg
        # print(self.assets)
        self.__solr = pysolr.Solr(solr_url) if solr_url else None

    def get(self, id: str):
        return self.assets.get(id)

    def search(self, *args, **kwargs):
        return self.__solr.search(*args, **kwargs)

    @property
    def default_up(self):
        return self.defaults.get('up')

    @property
    def default_front(self):
        return self.defaults.get('front')

    def _to_fullids(self, source: str, ids: list[str]):
        return [ id if '.' in id else f'{source}.{id}' for id in ids]

    def get_query_for_ids(self, ids: list[str]):
        fullids = self._to_fullids(self.config['source'], ids)
        return f'fullId:({" OR ".join(fullids)})'

    @staticmethod
    def _to_floats(s, default=None):
        return [float(x) for x in s.split(',')] if s else default

    @staticmethod
    def _to_scaled_floats(s, scale, default=None):
        return [float(x)*scale for x in s.split(',')] if s else default

    def _query_metadata(self, ids: list[str]):
        query = self.get_query_for_ids(ids)
        results = self.__solr.search(query, fl='fullId,wnsynsetkey,up,front,aligned.dims')
        converted = []
        if len(results):
            for result in results:
                up = result.get('up')
                up = AssetDb._to_floats(up, self.default_up)
                front = result.get('front')
                front = AssetDb._to_floats(front, self.default_front)
                raw_dims = result.get('aligned.dims')
                raw_dims = AssetDb._to_floats(raw_dims, None)
                aligned_dims = result.get('aligned.dims')
                aligned_dims = AssetDb._to_scaled_floats(aligned_dims, 1/100, None) 
                converted.append(EasyDict({
                    'fullId': result['fullId'],
                    'wnsynsetkey': result.get('wnsynsetkey', None),
                    'up': up,
                    'front': front,
                    'raw_dims': raw_dims,
                    'dims': aligned_dims }))
        return converted

    def get_metadata(self, id: str):
        results = self._query_metadata([id])
        if len(results) == 1:
            return results[0]
        return results

    def get_metadata_for_ids(self, ids: list[str]):
        return self._query_metadata(ids)

    # sort assets by closeness to specified dimensions
    def sort_by_dim(self, ids: list[str], dims: list[float]):
        metadata = self.get_metadata_for_ids(ids)
        for m in metadata:
            m['dim_se'] = np.sum((np.asarray(dims) - np.asarray(m['dims'])) ** 2)
        metadata = sorted(metadata, key=lambda m: m['dim_se'])
        return metadata
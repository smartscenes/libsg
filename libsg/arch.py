import base64
import copy
import struct
import sys
from pathlib import Path

from libsg.scene_types import ArchElement

class Architecture:
    HEADER = {
        'version': 'arch@1.0.2',
        'up': [0, 0, 1],
        'front': [0, 1, 0],
        'coords2d': [0, 1],
        'scaleToMeters': 1,
        'defaults': {
            'Wall': { 'depth': 0.1, 'extraHeight': 0.035 },
            'Ceiling': { 'depth': 0.05 },
            'Floor': { 'depth': 0.05}
        },
    }

    def __init__(self, id, imagedir=None):
        self.id = id
        self.imagedir = Path(imagedir) if imagedir is not None else None
        self.elements = []
        self.materials = []
        self.textures = []
        self.images = []
        self.regions = []
        self.is_typed = False

        self.__special_mats = {
            "Glass": { "uuid": "mat_glass", "color": "#777777", "opacity": 0.1, "transparent": True }
        }
        self.__included_special_mats = []


    def ensure_typed(self):
        # Make sure elements have correct type
        if not self.is_typed:
            self.elements = [ArchElement.from_json(element) for element in self.elements]
            self.is_typed = True

    def get_element_by_id(self, id):
        return next(self.find_elements(lambda elem: elem.id == id), None)

    def find_elements(self, cond):
        return filter(cond, self.elements)

    def find_elements_by_type(self, element_type):
        return filter(lambda elem: elem.type == element_type, self.elements)

    def add_region(self, region):
        self.regions.append(region)

    def add_element(self, element):
        self.elements.append(element)

    def add_elements(self, element):
        self.elements.extend(element)

    def create_material(self, element, name, flipY=True):
        id = element['id']
        imagefile = self.imagedir.joinpath(f'{id}.png')
        assert imagefile.suffix == '.png', "only PNG format supported currently"
        if not imagefile.exists():
            print(f'[Warning] image file {imagefile} not found, skipping texture creation.', file=sys.stderr)
            return
        element['materials'] = [ { 'name': name, 'materialId': f'mat_{id}' } ]
        self.materials.append({ 'uuid': f'mat_{id}', 'map': f'tex_{id}' })
        img_bytes = open(imagefile, "rb").read()
        blob = base64.b64encode(img_bytes).decode('ascii')
        width, height = struct.unpack('>LL', img_bytes[16:24])
        self.textures.append({ 'uuid': f'tex_{id}', 'repeat': [100 / width, 100 / height], 'image': f'img_{id}', 'flipY': flipY })
        self.images.append({ 'uuid': f'img_{id}', 'url': f'data:image/png;base64,{blob}' })

    def set_special_material(self, element, name, mat_name):        
        element['materials'] = [ { 'name': name, 'materialId': f'mat_{mat_name.lower()}' } ]
        if mat_name not in self.__included_special_mats:
            self.materials.append(self.__special_mats[mat_name])
            self.__included_special_mats.append(mat_name)

    def populate_materials(self, element):
        etype = element['type']
        if 'materials' in element and isinstance(element['materials'][0], str):
            name = "inside" if etype in ['Wall', 'Railing'] else "surface"
            self.set_special_material(element=element, name='inside', mat_name=element['materials'][0])
        elif self.imagedir is not None:
            if etype in ['Wall', 'Railing'] and element["height"] > 0:
                self.create_material(element=element, name='inside', flipY=True)
            elif etype in ['Ceiling', 'Floor', 'Landing', 'Ground']:
                self.create_material(element=element, name='surface', flipY=etype != 'Ceiling')
        if "railing" in element:
            for r in element["railing"]:
                self.populate_materials(r)


    def to_json(self):
        obj = copy.deepcopy(self.HEADER)
        obj['id'] = self.id
        obj['elements'] = [elem.to_json() for elem in self.elements] if self.is_typed else self.elements
        obj['regions'] = self.regions
        obj['materials'] = self.materials
        obj['textures'] = self.textures
        obj['images'] = self.images
        return obj

    @classmethod
    def from_json(cls, obj):
        arch = Architecture(obj['id'])
        arch.elements = obj['elements']
        arch.regions = obj['regions']
        arch.materials = obj['materials']
        arch.textures = obj['textures']
        arch.images = obj['images']
        arch.is_typed = False
        return arch

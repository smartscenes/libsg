def invert_mapping(mapping):
    inverted = {}
    for key, value in mapping.items():
        if value not in inverted:
            inverted[value] = [key]
        else:
            inverted[value].append(key)
    return inverted


# the mapping for this comes from ATISS, but the keys have been modified to match the spacing and capitalization used in
# solr
THREED_FRONT_BEDROOM_FURNITURE = {
    "Desk": "desk",
    "Nightstand": "nightstand",
    "King-size Bed": "double_bed",
    "Single bed": "single_bed",
    "Kids Bed": "kids_bed",
    "Ceiling Lamp": "ceiling_lamp",
    "Pendant Lamp": "pendant_lamp",
    "Bookcase / jewelry Armoire": "bookshelf",
    "TV Stand": "tv_stand",
    "Wardrobe": "wardrobe",
    "Lounge Chair / Cafe Chair / Office Chair": "chair",
    "Dining Chair": "chair",
    "Classic Chinese Chair": "chair",
    "armchair": "armchair",
    "Dressing Table": "dressing_table",
    "Dressing Chair": "dressing_chair",
    "Corner/Side Table": "table",
    "Dining Table": "table",
    "Round End Table": "table",
    "Drawer Chest / Corner cabinet": "cabinet",
    "Sideboard / Side Cabinet / Console Table": "cabinet",
    "Children Cabinet": "children_cabinet",
    "shelf": "shelf",
    "Footstool / Sofastool / Bed End Stool / Stool": "stool",
    "Coffee Table": "coffee_table",
    "Loveseat Sofa": "sofa",
    "Three-Seat / Multi-seat Sofa": "sofa",
    "L-shaped Sofa": "sofa",
    "Lazy Sofa": "sofa",
    "Chaise Longue Sofa": "sofa",
}
THREED_FRONT_BEDROOM_FURNITURE_INVERTED = invert_mapping(THREED_FRONT_BEDROOM_FURNITURE)

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "Bookcase / jewelry Armoire": "bookshelf",
    "Desk": "desk",
    "Pendant Lamp": "pendant_lamp",
    "Ceiling Lamp": "ceiling_lamp",
    "Lounge Chair / Cafe Chair / Office Chair": "lounge_chair",
    "Dining Chair": "dining_chair",
    "Dining Table": "dining_table",
    "Corner/Side Table": "corner_side_table",
    "Classic Chinese Chair": "chinese_chair",
    "armchair": "armchair",
    "Shelf": "shelf",
    "Sideboard / Side Cabinet / Console Table": "console_table",
    "Footstool / Sofastool / Bed End Stool / Stool": "stool",
    "Barstool": "stool",
    "Round End Table": "round_end_table",
    "Loveseat Sofa": "loveseat_sofa",
    "Drawer Chest / Corner cabinet": "cabinet",
    "Wardrobe": "wardrobe",
    "Three-Seat / Multi-seat Sofa": "multi_seat_sofa",
    "Wine Cabinet": "wine_cabinet",
    "Coffee Table": "coffee_table",
    "Lazy Sofa": "lazy_sofa",
    "Children Cabinet": "cabinet",
    "Chaise Longue Sofa": "chaise_longue_sofa",
    "L-shaped Sofa": "l_shaped_sofa",
    "TV Stand": "tv_stand",
}
THREED_FRONT_LIVINGROOM_FURNITURE_INVERTED = invert_mapping(THREED_FRONT_LIVINGROOM_FURNITURE)


label_to_3dfront_category = {
    "bedroom": THREED_FRONT_BEDROOM_FURNITURE_INVERTED,
    "livingroom": THREED_FRONT_LIVINGROOM_FURNITURE_INVERTED,
    "diningroom": THREED_FRONT_LIVINGROOM_FURNITURE_INVERTED,
}


def parse_bool_from_str(params, name, default: bool = None):
    raw = params.get(name)
    if raw is None:
        return default
    if raw.lower() not in {"true", "false"}:
        raise ValueError(f"Invalid value for {name}: {raw}")

    return raw.lower() == "true"

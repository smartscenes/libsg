import json
import logging
import re
from typing import Dict, Any, List, Tuple

import pandas as pd

from libsg.scene import Scene
from libsg.scene_types import SceneGraph
from libsg.llm import LLMSession

class SceneGraphMetric:
    FIELD_DEFINITIONS = {
        # "hasArchitecture": "A comma-separated list of architectural elements present in the scene.",
        # "hasOpening": "A list of architectural openings with their quantities and locations.",
        "hasEntity": "A list of objects (furniture, fixtures, etc.) present in the scene, along with their quantity specifications.",
        "hasAttribute": "A list of attributes associated with objects, openings, or architectural elements.",
        "hasRelationship": "A list of spatial or functional relationships between objects, openings, or architectural elements."
    }

    def __init__(self):
        """
        Initializes the SceneGraphMetric class.
        """
        self.num_scenes = 0
        self.evaluation_results: List[Dict[str, float]] = []
        self.unmatched_relationships = []
        
        logging.basicConfig(level=logging.DEBUG)

    def _evaluate_scene_graph(self, ground_truth: pd.Series, generated_scene_graph: SceneGraph) -> Dict[str, float]:
        """
        Evaluates the generated scene graph against the ground truth.
        If a field is not found in the ground truth, it is assumed to be 1.0

        Args:
            ground_truth (pd.Series): The ground truth data for the scene.
            generated_scene_graph (SceneGraph): The generated scene graph to be evaluated.

        Returns:
            Dict[str, float]: A dictionary with match percentages for each field.
        """
        results = {}
        print(f"Evaluating ground truth: {ground_truth.to_dict()}")
        for field in self.FIELD_DEFINITIONS:
            if field in ground_truth and pd.notna(ground_truth[field]):
                gt_value = ground_truth[field]
                gen_value = self._extract_field_from_scene_graph(generated_scene_graph, field)
                print(f"\nComparing {field}: \n\n Ground truth: {gt_value}, \n\n Generated: {gen_value}\n")
                match_percentage = getattr(self, f"_compare_{field}")(gt_value, gen_value)
                results[field] = match_percentage
            else:
                logging.warning(f"Field {field} not found in ground truth or is NaN")
                results[field] = 1.0
        return results

    def _extract_field_from_scene_graph(self, scene_graph: SceneGraph, field: str) -> List[str]:
        """
        Extracts the specified field from the scene graph.

        Args:
            scene_graph (SceneGraph): The scene graph from which to extract the field.
            field (str): The field to extract.

        Returns:
            List[str]: A list of extracted field values.
        """
        if field == "hasArchitecture":
            # return [obj.name for obj in scene_graph.objects if obj.name.lower() in ["bedroom", "closet", "bathroom", "kitchen", "living room"]]
            raise NotImplementedError("hasArchitecture is not implemented")
        elif field == "hasEntity":
            return [f"{obj.name} - quantityExact - 1" for obj in scene_graph.objects]
        elif field == "hasAttribute":
            return [f"{obj.name} - {attr}" for obj in scene_graph.objects for attr in obj.attributes]
        elif field == "hasRelationship":
            return [f"{rel.subject.name} - {rel.type} - {rel.target.name}" for rel in scene_graph.relationships]
        elif field == "hasOpening":
            # openings = []
            # for obj in scene_graph.objects:
            #     if obj.name.lower() in ["door", "window"]:
            #         location = next((attr for attr in obj.attributes if "in" in attr.lower()), "unknown location")
            #         openings.append(f"{obj.name} - quantityExact - 1 - {location}")
            # return openings
            raise NotImplementedError("hasOpening is not implemented")
        return []

    def _compare_has_opening(self, ground_truth: str, generated: List[str]) -> float:
        gt_openings = self._parse_openings(ground_truth)
        gen_openings = self._parse_openings("; ".join(generated))
        correct_count = 0
        total_count = sum(gt_info['quantity'] for gt_info in gt_openings.values())
        
        for opening, gt_info in gt_openings.items():
            for gen_opening, gen_info in gen_openings.items():
                if self._entity_match(opening, gen_opening):
                    if gt_info['quantity_type'] == gen_info['quantity_type']:
                        if gt_info['quantity_type'] == 'Exact':
                            correct_count += min(gt_info['quantity'], gen_info['quantity'])
                        else:  # AtLeast
                            correct_count += min(gt_info['quantity'], gen_info['quantity']) if gen_info['quantity'] >= gt_info['quantity'] else 0
                    if self._location_match(gt_info['location'], gen_info['location']):
                        correct_count += 1
        
        return correct_count / (total_count * 2) if total_count > 0 else 0.0  # Multiply by 2 because we're checking quantity and location
    
    def _parse_openings(self, openings_str: str) -> Dict[str, Dict[str, Any]]:
        openings = {}
        for opening in openings_str.split(";"):
            parts = opening.strip().split(" - ")
            if len(parts) >= 4:
                opening_type = parts[0]
                quantity_info = parts[1].split("quantity")[1]
                quantity_type, quantity = quantity_info.split(" - ")
                location = " - ".join(parts[3:])
                openings[opening_type] = {
                    'quantity_type': quantity_type,
                    'quantity': int(quantity),
                    'location': location
                }
        return openings

    def _location_match(self, gt_location: str, gen_location: str) -> bool:
        gt_location = gt_location.lower()
        gen_location = gen_location.lower()
        return gt_location in gen_location or gen_location in gt_location
    
    def _compare_has_architecture(self, ground_truth: str, generated: List[str]) -> float:
        """
        Compares the 'hasArchitecture' field between ground truth and generated data.

        Args:
            ground_truth (str): The ground truth value for the 'hasArchitecture' field.
            generated (List[str]): The generated values for the 'hasArchitecture' field.

        Returns:
            float: The match percentage.
        """
        gt_set = set(ground_truth.lower().split("; "))
        gen_set = set(item.lower() for item in generated)
        if not gen_set and "bedroom" in gt_set:
            # If no architecture is explicitly mentioned, but we're generating a bedroom, count it as correct
            gen_set.add("bedroom")
        intersection = gt_set.intersection(gen_set)
        return len(intersection) / max(len(gt_set), len(gen_set)) if max(len(gt_set), len(gen_set)) > 0 else 0.0

    def _compare_has_entity(self, ground_truth: str, generated: List[str]) -> float:
        """
        Compares the 'hasEntity' field between ground truth and generated data.
        Also handles duplicates in the generated entities.

        Args:
            ground_truth (str): The ground truth value for the 'hasEntity' field.
            generated (List[str]): The generated values for the 'hasEntity' field.

        Returns:
            float: The match percentage.
        """
        gt_entities = self._parse_entities(ground_truth)
        gen_entities = self._parse_entities("; ".join(generated))
        correct_count = 0
        total_count = sum(gt_info['quantity'] for gt_info in gt_entities.values())
        
        for entity, gt_info in gt_entities.items():
            matched_quantity = 0
            for gen_entity, gen_info in gen_entities.items():
                if self._entity_match(entity, gen_entity):
                    if gt_info['quantity_type'] == gen_info['quantity_type']:
                        if gt_info['quantity_type'] == 'Exact':
                            matched_quantity += gen_info['quantity']
                        else:  # AtLeast
                            matched_quantity += gen_info['quantity'] if gen_info['quantity'] >= gt_info['quantity'] else 0
            
            correct_count += min(matched_quantity, gt_info['quantity'])
        
        return correct_count / total_count if total_count > 0 else 0.0

    def _compare_has_attribute(self, ground_truth: str, generated: List[str]) -> float:
        """
        Compares the 'hasAttribute' field between ground truth and generated data.

        Args:
            ground_truth (str): The ground truth value for the 'hasAttribute' field.
            generated (List[str]): The generated values for the 'hasAttribute' field.

        Returns:
            float: The match percentage.
        """
        gt_attrs = self._parse_attributes(ground_truth)
        gen_attrs = self._parse_attributes("; ".join(generated))
        correct_count = 0
        total_attrs = sum(len(attrs) for attrs in gt_attrs.values())
        
        for gt_obj, gt_attr_list in gt_attrs.items():
            for gen_obj, gen_attr_list in gen_attrs.items():
                if self._entity_match(gt_obj, gen_obj):
                    for gt_attr in gt_attr_list:
                        if any(self._attribute_match(gt_attr, gen_attr) for gen_attr in gen_attr_list):
                            correct_count += 1
        
        return correct_count / total_attrs if total_attrs > 0 else 0.0

    def _compare_has_relationship(self, ground_truth: str, generated: List[str]) -> float:
        """
        Compares the 'hasRelationship' field between ground truth and generated data.
        Handles "in" relationships separately and removes them before comparing others.
        Keeps track of unmatched relationships.

        Args:
            ground_truth (str): The ground truth value for the 'hasRelationship' field.
            generated (List[str]): The generated values for the 'hasRelationship' field.

        Returns:
            float: The match percentage.
        """
        gt_rels = self._parse_relationships(ground_truth)
        gen_rels = self._parse_relationships("; ".join(generated))
        correct_count = 0
        total_count = len(gt_rels)
        
        unmatched_relationships = []
        
        # Handle "in" relationships first
        in_relationships = [rel for rel in gt_rels if rel[1].lower() == "in"]
        for in_rel in in_relationships:
            if any(self._entity_match(in_rel[0], gen_rel[0]) for gen_rel in gen_rels):
                correct_count += 1
            else:
                unmatched_relationships.append(in_rel)
            gt_rels.remove(in_rel)
        
        # Compare remaining relationships
        for gt_rel in gt_rels:
            matched = False
            for gen_rel in gen_rels:
                if (self._entity_match(gt_rel[0], gen_rel[0]) and
                    self._relationship_match(gt_rel[1], gen_rel[1]) and
                    self._entity_match(gt_rel[2], gen_rel[2])):
                    correct_count += 1
                    matched = True
                    break
            if not matched:
                unmatched_relationships.append(gt_rel)
        
        # Store unmatched relationships in an instance variable
        self.unmatched_relationships = unmatched_relationships
        print(f"Unmatched relationships: {unmatched_relationships}")
        
        return correct_count / total_count if total_count > 0 else 1.0

    def _parse_entities(self, entities_str: str) -> Dict[str, Dict[str, Any]]:
        """
        Parses the entities from a string.

        Args:
            entities_str (str): The string containing entities.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of parsed entities.
        """
        entities = {}
        for entity in entities_str.split(";"):
            parts = entity.strip().split(" - ")
            if len(parts) == 3:
                entity_name = parts[0]
                quantity_type = parts[1].split("quantity")[1]
                quantity = int(parts[2])
                entities[entity_name] = {
                    'quantity_type': quantity_type,
                    'quantity': quantity
                }
        return entities

    def _attribute_match(self, gt_attr: str, gen_attr: str) -> bool:
        """
        Checks if two attributes match.

        Args:
            gt_attr (str): The ground truth attribute.
            gen_attr (str): The generated attribute.

        Returns:
            bool: True if the attributes match, False otherwise.
        """
        gt_attr = gt_attr.lower()
        gen_attr = gen_attr.lower()
        return gt_attr in gen_attr or gen_attr in gt_attr or self._are_synonyms(gt_attr, gen_attr)
    
    def _parse_attributes(self, attributes_str: str) -> Dict[str, List[str]]:
        """
        Parses the attributes from a string.

        Args:
            attributes_str (str): The string containing attributes.

        Returns:
            Dict[str, List[str]]: A dictionary of parsed attributes.
        """
        attrs = {}
        for item in attributes_str.split(";"):
            parts = item.strip().split(" - ", 1)
            if len(parts) == 2:
                obj, attr = parts
                if obj not in attrs:
                    attrs[obj] = []
                attrs[obj].append(attr)
        return attrs

    def _parse_relationships(self, relationships_str: str) -> List[Tuple[str, str, str]]:
        """
        Parses the relationships from a string.

        Args:
            relationships_str (str): The string containing relationships.

        Returns:
            List[Tuple[str, str, str]]: A list of parsed relationships.
        """
        return [tuple(item.strip().split(" - ")) for item in relationships_str.split(";") if " - " in item]
    
    def _entity_match(self, gt_entity: str, gen_entity: str) -> bool:
        """
        Checks if two entities match, considering synonyms and partial matches.

        Args:
            gt_entity (str): The ground truth entity.
            gen_entity (str): The generated entity.

        Returns:
            bool: True if the entities match, False otherwise.
        """
        # print(f"Comparing {gt_entity} and {gen_entity}")
        gt_entity = gt_entity.lower()
        gen_entity = gen_entity.lower()
        
        # Remove any trailing numbers from the ground truth entity
        if "_" in gt_entity:
            gt_entity = gt_entity.split("_")[0]
        
        # Check for exact match, partial match, or synonyms
        return (gt_entity in gen_entity or 
                gen_entity in gt_entity or 
                self._are_synonyms(gt_entity, gen_entity) or 
                self._partial_match(gt_entity, gen_entity))

    def _partial_match(self, gt_entity: str, gen_entity: str) -> bool:
        """
        Checks if the generated entity partially matches the ground truth entity.

        Args:
            gt_entity (str): The ground truth entity.
            gen_entity (str): The generated entity.

        Returns:
            bool: True if there is a partial match, False otherwise.
        """
        # print(f"Checking partial match for {gt_entity} and {gen_entity}")
        gt_words = gt_entity.split()
        gen_words = gen_entity.split()
        result = any(gt_word in gen_words for gt_word in gt_words)
        if result:
            print(f"Partial match found: {gt_entity} in {gen_entity}")
        return result

    def _attribute_match(self, gt_attr: str, gen_attr: str) -> bool:
        """
        Checks if two attributes match, considering synonyms.

        Args:
            gt_attr (str): The ground truth attribute.
            gen_attr (str): The generated attribute.

        Returns:
            bool: True if the attributes match, False otherwise.
        """
        gt_attr = gt_attr.lower()
        gen_attr = gen_attr.lower()
        return gt_attr in gen_attr or gen_attr in gt_attr or self._are_synonyms(gt_attr, gen_attr)

    def _relationship_match(self, gt_rel: str, gen_rel: str) -> bool:
        """
        Checks if two relationships match after removing unnecessary words.
        Remove "closely" and "of" in the generated relationship
        
        Args:
            gt_rel (str): The ground truth relationship.
            gen_rel (str): The generated relationship.

        Returns:
            bool: True if the relationships match, False otherwise.
        """
        pattern = r'\b(closely|of)\b'
        gt_rel = gt_rel.lower()
        gen_rel = re.sub(pattern, '', gen_rel.lower()).strip()
        
        return gt_rel == gen_rel

    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """
        Checks if two words are synonyms.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            bool: True if the words are synonyms, False otherwise.
        """
        synonyms = {
            "bed": ["single bed", "twin-sized bed", "queen-size bed", "king-size bed"],
            "tv": ["television", "TV"],
            "wardrobe": ["closet", "cabinet"],
            "nightstand": ["bedside table", "side table"]
        }
        return any(word1 in syns and word2 in syns for syns in synonyms.values())
    
    def _cleanup_scene_graph(self, scene_graph: SceneGraph) -> SceneGraph:
        """
        Clean up the generated scene graph using LLM.
        Remove any attribute in the name field and move it to the attributes field, then return the full scene graph in json format

        Args:
            scene_graph (SceneGraph): The original generated scene graph.

        Returns:
            SceneGraph: The cleaned up scene graph.
        """
        llm = LLMSession(log_tasks=False, log_responses=False)
        
        # Convert scene graph to a string representation
        scene_graph_str = scene_graph.to_json()

        # Use LLM to clean up the scene graph
        prompt = f"""you are required to remove any attribute in the name field and move it to the attributes field, then return the full scene graph in json format
                     e.g. "double bed" into "bed":\n\n{scene_graph_str}"""
        cleaned_scene_graph = llm.send(prompt, is_json=True)

        # Convert the cleaned up string back to a SceneGraph object
        cleaned_scene_graph = SceneGraph.from_json(cleaned_scene_graph)

        return cleaned_scene_graph
    
    def __call__(self, inp: str, scene_graph: SceneGraph, scene: Scene, **kwargs):
        """
        Evaluates the scene graph based on the input prompt and scene data.

        Args:
            inp (str): The input prompt.
            scene_graph (SceneGraph): The generated scene graph.
            scene (Scene): The scene data.
            prompt_data (pd.DataFrame): The prompt data containing ground truth.

        Returns:
            None
        """
        prompt_data = kwargs.get("prompt_data")
        logging.info("Evaluating scene graph")
        matching_prompt = prompt_data[prompt_data['prompt'] == inp]
        if matching_prompt.empty:
            logging.warning(f"No matching prompt found for: {inp}")
            return
        
        # Clean up the scene graph using LLM
        cleaned_scene_graph = self._cleanup_scene_graph(scene_graph)

        ground_truth = matching_prompt.iloc[0]
        match_percentages = self._evaluate_scene_graph(ground_truth, cleaned_scene_graph)
        self.evaluation_results.append(match_percentages)
        self.num_scenes += 1
        logging.info(f"Evaluation results: {match_percentages}")

    def log(self) -> Dict[str, float]:
        """
        Logs the average match percentages for all evaluated scenes.

        Returns:
            Dict[str, float]: A dictionary with average match percentages for each field.
        """
        if self.num_scenes == 0:
            logging.warning("No scenes evaluated")
            return {field: 0.0 for field in self.FIELD_DEFINITIONS}

        average_matches = {}
        for field in self.FIELD_DEFINITIONS:
            field_matches = [result[field] for result in self.evaluation_results if field in result]
            average_matches[field] = sum(field_matches) / len(field_matches) if field_matches else 0.0

        for field, avg_match in average_matches.items():
            logging.info(f"Scene Graph metrics - {field}: Average match percentage: {avg_match:.2f}")

        return average_matches
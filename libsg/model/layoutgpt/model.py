import json
import logging
import os
import time
from string import digits

import numpy as np
import torch
from openai import OpenAI
from transformers import GPT2TokenizerFast

from libsg.model.layoutgpt.examples import load_examples, load_features


class LayoutGPT:
    """Reimplementation of LayoutGPT for 3D scenes, as in https://github.com/weixi-feng/LayoutGPT.git."""

    def __init__(
        self,
        model_name: str,
        room_type: str,
        examples_path: str,
        stats_file: str,
        id_path: str,
        unit: str = "px",
        gpt_input_length_limit: int = 7000,
        icl_type: str = "k-similar",
        top_k: int = 8,
        temperature: float = 0.7,
        top_p: float = 1.0,
        normalize: bool = True,
        scaling_factor: int = 256,
        **kwargs,
    ):
        self.model_name = model_name
        self.room_type = room_type
        self.unit = unit  # px, m, or ''
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.max_input_length = gpt_input_length_limit
        self.icl_type = icl_type  # fixed-random or k-similar
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.normalize = normalize
        self.scaling_factor = scaling_factor

        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )

        self.client = OpenAI(api_key=api_key)

        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        with open(id_path, "r") as f:
            splits = json.load(f)
        ids = splits["train"]
        self.reference_examples, metadata = load_examples(
            examples_path, ids, self.stats, self.unit, self.room_type, normalize
        )
        self.reference_features = load_features(metadata)

    @staticmethod
    def _get_room_dims(room_mask: np.ndarray):
        occupied = np.where(room_mask)
        room_length = np.max(occupied[0]) - np.min(occupied[0])
        room_width = np.max(occupied[1]) - np.min(occupied[1])

        return room_length, room_width

    def craft_input(self, raw_input: str, room_mask: np.ndarray, room_type: str = None, scale_factor: int = 256):
        room_type = room_type if room_type is not None else self.room_type

        condition = f"Condition:\n"
        if room_type == "livingroom":
            condition += f"Room Type: living room\n"
        elif room_type == "diningroom":
            condition += f"Room Type: dining room\n"
        else:
            condition += f"Room Type: {room_type}\n"

        # get room length and width
        room_length, room_width = self._get_room_dims(room_mask)
        if self.normalize:
            norm = min(room_length, room_width)
            room_length, room_width = room_length / norm, room_width / norm
            if self.unit in ["px", ""]:
                room_length, room_width = int(room_length * scale_factor), int(room_width * scale_factor)

        condition += f"Room Size: max length {room_length}{self.unit}, max width {room_width}{self.unit}\n"

        # add text description (NOT in the original LayoutGPT)
        if raw_input is not None:
            condition += f"Description: {raw_input}\n"

        return condition

    def get_closest_room(self, reference_features: np.ndarray, query: np.ndarray):
        """
        train_features
        """
        distances = [[id, ((feat - query) ** 2).mean()] for id, feat in reference_features.items()]
        distances = sorted(distances, key=lambda x: x[1])
        sorted_ids, _ = zip(*distances)
        return sorted_ids

    def get_supporting_examples(self, room_mask: np.ndarray):
        if self.icl_type == "fixed-random" or room_mask is None:
            return list(self.reference_examples.values())[: self.top_k]
        if self.icl_type == "k-similar":
            sorted_ids = self.get_closest_room(self.reference_features, room_mask)
            return [self.reference_examples[id] for id in sorted_ids[: self.top_k]]

    def form_prompt_for_chatgpt(self, text_input, supporting_examples):
        message_list = []
        unit_name = "pixel" if self.unit in ["px", ""] else "meters"
        class_freq = [f"{obj}: {round(self.stats['class_frequencies'][obj], 4)}" for obj in self.stats["object_types"]]
        rtn_prompt = (
            "You are a 3D indoor scene designer. \nInstruction: synthesize the 3D layout of an indoor scene. "
            "The generated 3D layout should follow the CSS style, where each line starts with the furniture category "
            "and is followed by the 3D size, orientation and absolute position. "
            "Formally, each line should follow the template: \n"
            f"FURNITURE {{length: ?{self.unit}: width: ?{self.unit}; height: ?{self.unit}; orientation: ? degrees; left: ?{self.unit}; top: ?{self.unit}; depth: ?{self.unit};}}\n"
            f"All values are in {unit_name} but the orientation angle is in degrees.\n\n"
            f"Available furnitures: {', '.join(self.stats['object_types'])} \n"
            f"Overall furniture frequencies: ({'; '.join(class_freq)})\n\n"
        )

        message_list.append({"role": "system", "content": rtn_prompt})
        last_example = f"{text_input[0]}Layout:\n"
        total_length = len(self.tokenizer(rtn_prompt + last_example)["input_ids"])

        # loop through the related supporting examples, check if the prompt length exceed limit
        for i, supporting_example in enumerate(supporting_examples[: self.top_k]):
            cur_len = len(self.tokenizer(supporting_example[0] + supporting_example[1])["input_ids"])
            if total_length + cur_len > self.max_input_length:  # won't take the input that is too long
                logging.debug(f"{i+1}th exemplar exceed max length")
                break
            total_length += cur_len

            current_messages = [
                {"role": "user", "content": supporting_example[0] + "Layout:\n"},
                {"role": "assistant", "content": supporting_example[1].lstrip("Layout:\n")},
            ]
            message_list = message_list + current_messages

        # concatename prompts for gpt4
        message_list.append({"role": "user", "content": last_example})

        return message_list

    def make_request(self, prompts: list[dict], room_type: str = None):
        if room_type is None:
            room_type = self.room_type
        # try:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=prompts,
            temperature=self.temperature,
            max_tokens=1024 if room_type == "livingroom" else 512,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop="Condition:",
            n=1,
        )
        # except openai.error.ServiceUnavailableError:
        #     print("OpenAI ServiceUnavailableError.\tWill try again in 5 seconds.")
        #     time.sleep(5)
        # except openai.error.RateLimitError:
        #     print("OpenAI RateLimitError.\tWill try again in 5 seconds.")
        #     time.sleep(5)
        # except openai.error.InvalidRequestError as e:
        #     print(e)
        #     print("Input too long. Will shrink the prompting examples.")
        #     top_k -= 1
        # except openai.error.APIError as e:
        #     print("OpenAI Bad Gateway Error.\tWill try again in 5 seconds.")
        #     time.sleep(5)

    def parse_layout(self, raw_output: str):
        # cannot use cssutils due to self-defined properties
        try:
            text, bbox = raw_output.split("{")
            bbox = bbox.strip().strip("}").strip().strip(";").split(";")
            assert len(bbox) == 7
            bbox = [b.strip().strip(";").rstrip("degrees").strip() for b in bbox]
        except:
            return None, None

        category = text.strip()
        parsed_category = category.translate(category.maketrans("", "", digits)).strip()

        bbox = [b.split(":") for b in bbox]

        bbox = {k.strip(): float(v.lstrip().rstrip(self.unit)) for k, v in bbox}
        if sorted(bbox.keys()) != sorted(["height", "width", "length", "orientation", "left", "top", "depth"]):
            bbox = {k: 0 for k in bbox.keys()}
            return parsed_category, bbox

        return parsed_category, bbox

    def generate_boxes(self, raw_input: str, room_mask: np.ndarray = None, room_type: str = None):
        # form prompt
        room_mask = room_mask.cpu().numpy()[0, 0, :, :]
        room_length, room_width = self._get_room_dims(room_mask)
        norm = min(room_length, room_width)
        if self.unit in ["px", ""]:
            norm /= self.scaling_factor

        supporting_examples = self.get_supporting_examples(room_mask)
        inp = self.craft_input(raw_input, room_mask, room_type, scale_factor=self.scaling_factor)
        print("Input:\n", inp)
        prompts = self.form_prompt_for_chatgpt(text_input=inp, supporting_examples=supporting_examples)

        # run LLM
        response = self.make_request(prompts, room_type=room_type)

        # parse output
        line_list = response.choices[0].message.content.split("\n")

        output = {"class_labels": [], "sizes": [], "angles": [], "translations": []}
        for line in line_list:
            if line == "":
                continue
            try:
                selector_text, bbox = self.parse_layout(line)
                if selector_text is None:
                    continue
                print(f"{selector_text}: {', '.join([f'{k}: {v}' for k, v in bbox.items()])}")
                output["class_labels"].append(selector_text)
                output["sizes"].append(
                    [
                        bbox["length"] * norm,
                        bbox["width"] * norm,
                        bbox["height"] * norm,
                    ]
                )
                output["angles"].append([bbox["orientation"]])
                output["translations"].append(
                    [
                        bbox["left"] * norm,
                        bbox["top"] * norm,
                        bbox["depth"] * norm,
                    ]
                )
            except ValueError:
                pass

        # convert to tensor and add batch dimension
        output["class_labels"] = [output["class_labels"]]
        output["sizes"] = torch.tensor(output["sizes"]).unsqueeze(0)
        output["angles"] = torch.tensor(output["angles"]).unsqueeze(0)
        output["translations"] = torch.tensor(output["translations"]).unsqueeze(0)

        return output

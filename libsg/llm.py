import base64
import logging
import os
from typing import Callable

from openai import OpenAI
import json

# from libsg.config import config as cfg

class LLMSession:
    def __init__(self, system_prompt: str|None = None, model: str = "gpt-4o-2024-08-06", log_tasks: bool = True, log_responses: bool = False) -> None:
        '''
        Initialize a LLM_Session.

        Args:
            system_prompt: string, the system prompt for the LLM
            print_tasks: boolean, whether to print the tasks to the console
            print_responses: boolean, whether to print the responses to the console

        Returns:
            None
        '''

        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )

        client = OpenAI(api_key=api_key)
        
        self.client = client
        self.model = model
        self.past_tasks: list[str] = []
        self.past_messages = [{"role": "system", "content": system_prompt}] if system_prompt is not None else []
        self.past_responses: list[str] = []
    
        self.print_tasks = log_tasks
        self.print_responses = log_responses
    
    def send(self, task: str, prompt_info: dict[str, str] | None = None, is_json: bool = False) -> str:
        '''
        Send a message of a specific task to the GPT-4 model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            is_json: boolean, whether the response should be in JSON format
        Returns:
            response: string, the response from the model
        '''

        if self.print_tasks:
            logging.debug(f"$ --- LLM --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        self._send(prompt, is_json)
        response = self.past_responses[-1]
        if self.print_responses:
            logging.debug(f"$ --- LLM --- Response:\n{response}\n")

        return response
    
    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None) -> str:
        '''
        Make a prompt for the LLM model.

        Args:
            task: string, the task of the prompt
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info
            
        Returns:
            prompt: string, the prompt for the LLM model
        '''

        # Get the predefined prompt for the task
        # prompt = self.predefined_prompts[task]

        prompt = task
        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", prompt_info[key])

        return prompt
    
    def send(self, task: str, prompt_info: dict[str, str] | None = None, is_json: bool = False) -> str | dict | list:
        '''
        Send a message of a specific task to the GPT-4 model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            is_json: boolean, whether the response should be in JSON format
        Returns:
            response: string, the response from the model or a dict/list if is_json is True
        '''

        if self.print_tasks:
            logging.debug(f"$ --- LLM --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        self._send(prompt, is_json)
        response = self.past_responses[-1]

        if is_json:
            response = json.loads(response)

        return response
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _send(self, new_message: str, is_json: bool = False, temperature: float = 0.7, image_path: str | None = None) -> None:
        '''
        Send a message to the GPT-4 model along with the past messages and store the response.

        Args:
            new_message: string, the new message to be sent to the model
            is_json: boolean, whether the response should be in JSON format
            temperature: float, the temperature of the model
            image_path: string, the path to the image to be sent to the model
        
        Returns:
            None
        '''

        if image_path is not None:
            image_base64 = self._encode_image(image_path)
            new_message = f"{new_message}\n\n![alt text](data:image/jpeg;base64,{image_base64})"

        self.past_messages.append({"role": "user", "content": new_message})
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.past_messages,
            response_format={ "type": "json_object" } if is_json else None,
            temperature=temperature
        )
        
        response = completion.choices[0].message.content
        self.past_messages.append({"role": "assistant", "content": response})
        self.past_responses.append(response)
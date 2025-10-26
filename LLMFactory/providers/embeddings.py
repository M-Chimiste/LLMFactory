# Copyright 2023 M Chimiste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sentence transformer embedding inference provider."""

from typing import List, Union, Optional

from .base import InferenceModel


class SentenceTransformerInference(InferenceModel):
    def __init__(self,
                 model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
                 remote_code: bool = True,
                 device: Optional[str] = None):
        """
        Initializes a SentenceTransformerInference model instance.

        This constructor sets up the SentenceTransformer model for inference tasks. It allows for customization of the model name, remote code execution, and device selection.

        Args:
            model_name (str, optional): The name of the SentenceTransformer model to use. Defaults to "Alibaba-NLP/gte-large-en-v1.5".
            remote_code (bool, optional): Whether to allow remote code execution. Defaults to True.
            device (Optional[str], optional): The device to use for model computations. Defaults to None, which automatically selects the best available device.

        Raises:
            ValueError: If an invalid device is specified.
        """
        import torch
        self.remote_code = remote_code
        if not device:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        super().__init__(model_name)

    def _get_provider(self) -> str:
        return "sentence-transformer"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name, trust_remote_code=self.remote_code, device=self.device)

    def invoke(self,
               text: Union[str, List[str]],
               *,
               streaming: bool = False,
               to_list: bool = False,
               normalize: bool = False,
               batch_size: Optional[int] = None,
               show_progress_bar: Optional[bool] = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               **kwargs) -> Union[List, object]:
        """
        Invokes the SentenceTransformer model to generate embeddings for the given text(s).

        This method processes the input text(s) and returns their corresponding embeddings. It supports
        various options for customization, such as streaming, converting to list, normalization, and more.

        Args:
            text (Union[str, List[str]]): The input text or list of texts to generate embeddings for.
            streaming (bool, optional): If True, processes the input text in a streaming fashion. Defaults to False.
            to_list (bool, optional): If True, converts the output embeddings to a list. Defaults to False.
            normalize (bool, optional): If True, normalizes the embeddings. Defaults to False.
            batch_size (Optional[int], optional): The batch size to use for encoding. Defaults to None.
            show_progress_bar (Optional[bool], optional): If True, shows a progress bar during encoding. Defaults to None.
            convert_to_numpy (bool, optional): If True, converts the embeddings to numpy arrays. Defaults to True.
            convert_to_tensor (bool, optional): If True, converts the embeddings to tensors. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the encoding function.

        Returns:
            Union[List, object]: The generated embeddings. If `to_list` is True, returns a list of embeddings. Otherwise, returns the embeddings as is.
        """
        _ = streaming
        is_string_input = isinstance(text, str)
        if is_string_input:
            text = [text]

        # Build encode parameters
        encode_params = {
            'sentences': text,
            'normalize_embeddings': normalize,
            'convert_to_numpy': convert_to_numpy,
            'convert_to_tensor': convert_to_tensor,
            **kwargs
        }

        # Add optional parameters if provided
        if batch_size is not None:
            encode_params['batch_size'] = batch_size
        if show_progress_bar is not None:
            encode_params['show_progress_bar'] = show_progress_bar

        embeddings = self.client.encode(**encode_params)

        if to_list and convert_to_numpy:
            embeddings = [embedding.tolist() for embedding in embeddings]

        # Only return single embedding if input was a single string, not a list
        if is_string_input and len(embeddings) == 1:
            return embeddings[0]
        return embeddings

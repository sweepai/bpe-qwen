from typing import List, Dict, Any, Union, Optional
from pathlib import Path

class QwenTokenizer:
    def __init__(self, tokenizer_dir: str = ...) -> None: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
    def __call__(self, texts: Union[str, List[str]], **kwargs: Any) -> Dict[str, List[List[int]]]: ...
    def vocab_size(self) -> int: ...

class QwenTokenizerFast:
    model_max_length: int
    padding_side: str
    pad_token: str
    pad_token_id: int
    eos_token: str
    eos_token_id: int
    unk_token: Optional[str]
    unk_token_id: Optional[int]
    bos_token: Optional[str]
    bos_token_id: Optional[int]
    add_special_tokens: bool
    clean_up_tokenization_spaces: bool
    vocab_size: int

    def __init__(self,
                 vocab_file: Optional[str] = None,
                 merges_file: Optional[str] = None,
                 tokenizer_file: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 **kwargs: Any) -> None: ...

    def encode(self,
               text: str,
               add_special_tokens: Optional[bool] = None,
               padding: Union[bool, str] = False,
               truncation: Union[bool, str] = False,
               max_length: Optional[int] = None,
               return_tensors: Optional[str] = None,
               **kwargs: Any) -> Union[List[int], Any]: ...

    def decode(self,
               token_ids: Union[List[int], Any],
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: Optional[bool] = None,
               **kwargs: Any) -> str: ...

    def __call__(self,
                 text: Union[str, List[str]],
                 add_special_tokens: Optional[bool] = None,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None,
                 return_attention_mask: bool = False,
                 **kwargs: Any) -> Dict[str, Any]: ...

    def batch_encode_plus(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def encode_plus(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def tokenize(self, text: str, **kwargs: Any) -> List[str]: ...
    def get_vocab(self) -> Dict[str, int]: ...
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None: ...

def patch_transformers() -> bool: ...
def unpatch_transformers() -> bool: ...

__all__ = ["QwenTokenizer", "QwenTokenizerFast", "patch_transformers", "unpatch_transformers"]
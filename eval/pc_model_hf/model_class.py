import torch
import pyjuice as juice
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

class PyJuiceHubModel(PyTorchModelHubMixin):
    def __init__(self, config=None):
        self.config = config
        self.pc = None

    @classmethod
    def _from_pretrained(cls, model_id, revision, cache_dir, force_download, proxies, resume_download, local_files_only, token, **model_kwargs):
        model_path = hf_hub_download(repo_id=model_id, filename="model.jpc")
        
        pc = juice.load(model_path)
        
        instance = cls()
        instance.pc = pc
        return instance
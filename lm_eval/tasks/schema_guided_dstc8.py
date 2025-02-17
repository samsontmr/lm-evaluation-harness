"""
Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset
https://arxiv.org/abs/1909.05855

Multi-domain, task-oriented conversations created for the DSTC8 challenge.
Here, the dataset is be used for evaluating response generation.

Homepage: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
"""
from lm_eval.base import PromptSourceTask, rf
from lm_eval import metrics

_CITATION = """
@inproceedings{rastogi2020towards,
  title={Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset},
  author={Rastogi, Abhinav and Zang, Xiaoxue and Sunkara, Srinivas and Gupta, Raghav and Khaitan, Pranav},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={8689--8696},
  year={2020}
}
"""

class Schema_Guided_DSTC8(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "schema_guided_dstc8"
    DATASET_NAME = "dialogues"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
    
    def max_generation_length(self):
        return 64

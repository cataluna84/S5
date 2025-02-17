"""Long Range Arena datasets"""
import io
import logging
import os
import pickle
from pathlib import Path

from collections import Counter, OrderedDict
from typing import Dict, List, Optional
import torch
from torch import nn
import torch.nn.functional as F

from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.trainers import WordLevelTrainer

import torchvision
from einops.layers.torch import Rearrange, Reduce
from PIL import Image  # Only used for Pathfinder
from datasets import DatasetDict, Value, load_dataset

from .base import default_data_path, SequenceDataset, ImageResolutionSequenceDataset
import pandas as pd 


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()

def create_vocab_from_iterable(
    iterable, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True
) :
    r"""Code which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurance frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.

    Returns:
        list: A list of tokens
        int: The size of the vocabulary without the eventual special tokens
    Examples:
 
    """
    counter = Counter()
    for tokens in iterable:
        counter.update(tokens)
    # First sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    # TODO (louis): I am not certain if the vocab size should include special
    # tokens or not,this line does not include them, just in case you need it
    #vocab_size = len(ordered_dict)

    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = []
    # Save room for special tokens
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)

    if special_first:
        tokens[0:0] = specials
    else:
        tokens.extend(specials)

    return {token:id for id, token in enumerate(tokens)}

def tokens_to_indices(tokens, vocab):
    """
    Converts a list of tokens into a list of their corresponding indices in the vocabulary.

    Args:
        tokens (list): A list of tokens (strings).
        vocab (dict): A dictionary mapping tokens to their indices.

    Returns:
        list: A list of indices corresponding to the input tokens.
    """
    # TODO: (louis) Could be more elegant and using the unknow token 
    return [vocab[token] if token in vocab.keys() else -1 for token in tokens]


class IMDB(SequenceDataset):
    _name_ = "imdb"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        lengths = torch.tensor([len(x) for x in xs])
        if self.level == "char":
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab['<pad>'], batch_first=True
            )

            print(f'{xs=}')
            print(f'{xs.shape=}')
        else : 
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.tokenizer.token_to_id("[PAD]"), batch_first=True
            )
        
        ys = torch.tensor(ys)
        return xs, ys, {"lengths": lengths}

        # self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
            )
            trainer = WordLevelTrainer(min_frequency = self.min_freq, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer)
            tokenizer.enable_truncation(max_length=l_max)
            tokenize = lambda example: {'input_ids' : tokenizer.encode(example["text"]).ids}
            dataset = dataset.map(
                tokenize,
                remove_columns=["text"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=max(self.n_workers, 1),
            )
            vocab = tokenizer.get_vocab()
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars
            # Account for <bos> and <eos> tokens
            tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}

            dataset = dataset.map(
                tokenize,
                remove_columns=["text"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=max(self.n_workers, 1),
            )

            special_tokens = (
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            )
            vocab = create_vocab_from_iterable(dataset['train']["tokens"], specials=special_tokens)

            numericalize = lambda example: {
                "input_ids": tokens_to_indices(
                    (["<bos>"] if self.append_bos else [])
                    + example["tokens"]
                    + (["<eos>"] if self.append_eos else []), 
                    vocab=vocab,
                )
            }

            dataset = dataset.map(
                numericalize,
                remove_columns=["tokens"],
                keep_in_memory=True,
                load_from_cache_file=False,
                num_proc=max(self.n_workers, 1),
            )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        os.makedirs(cache_dir, exist_ok=True)  # Create the directory if it doesn't exist
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        if self.level == "char": # The tokenizer is a vanilla list tokenizer
            with open(cache_dir / "tokenizer.pkl", "wb") as f:
                pickle.dump(tokenizer, f)
            with open(cache_dir / "vocab.pkl", "wb") as g:
                pickle.dump(vocab, g)    
        else : # The tokenizer is a HF tokenizer
            tokenizer.save(str(cache_dir / "tokenizer.json"))
        dataset.save_to_disk(str(cache_dir))
        
    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        if self.level == "word":
            tokenizer = Tokenizer.from_file(str(cache_dir/"tokenizer.json"))
            vocab = tokenizer.get_vocab()
        else : 
            with open(cache_dir / "tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            with open(cache_dir / "vocab.pkl", "rb") as f:
                vocab = pickle.load(f)

        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

import pandas as pd

class TabularDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        format,
        col_idx=None,
        skip_header=False,
        csv_reader_params=None,
    ):
        """
        col_idx: the indices of the columns.
        """
        # ... existing code ...

        format = format.lower()
        assert format in ["tsv", "csv"]
        
        if format == "csv":
            self._data = pd.read_csv(os.path.expanduser(path), **csv_reader_params)
        elif format == "tsv":
            self._data = pd.read_csv(os.path.expanduser(path), sep="\t", **csv_reader_params)
        
        if skip_header:
            self._data = self._data.iloc[1:]
        
        if col_idx is not None:
            self._data = self._data.iloc[:, col_idx]

        self._data = self._data.values.tolist()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]



class ListOps(SequenceDataset):
    _name_ = "listops"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        

        special_tokens = (
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            )
        
        vocab = create_vocab_from_iterable(
            dataset["train"]["tokens"], 
            specials=special_tokens
            )
        
        self.pad_token_id = 0

        numericalize = lambda example: {
            "input_ids": tokens_to_indices(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else []), 
                vocab=vocab,
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer,vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        os.makedirs(cache_dir, exist_ok=True)  # Create the directory if it doesn't exist
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class PathFinder(ImageResolutionSequenceDataset):
    _name_ = "pathfinder"
    d_input = 1
    d_output = 2
    l_output = 0

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 32,
            "sequential": True,
            "tokenize": False,
            "center": True,
            "pool": 1,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
        }

    def default_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            if self.center:
                transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize
                else Rearrange("1 h w -> (h w) 1")
            )
        else:
            transform_list.append(Rearrange("1 h w -> h w 1"))
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is either 32, 64, 128, or 256.
            """
            )

    def setup(self, stage=None):
        if self.data_dir is None:
            self.data_dir = (
                default_data_path / self._name_ / f"pathfinder{self.resolution}"
            )

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

class AAN(SequenceDataset):
    _name_ = "aan"
    d_output = 2  # Use accuracy instead of binary_accuracy
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the tsv_data directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            # Pad both to same length
            # Shape (batch, length)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L-xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L-xs2.size(1)), value=self.vocab["<pad>"])
            ys = torch.tensor(ys)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )  # True)
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        self.pad_token_id = 0

        special_tokens=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            )
        
        vocab = create_vocab_from_iterable(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"], 
            specials=special_tokens)
        
    
        encode = lambda text: tokens_to_indices(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else []),
            vocab=vocab,
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer,vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        os.makedirs(cache_dir, exist_ok=True)  # Create the directory if it doesn't exist
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

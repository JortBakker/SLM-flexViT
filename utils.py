from typing import Union, Any
import shutil
import os
import io

from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop, ConvertImageDtype, RandAugment
)
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
import tqdm
from timm.data import Mixup
from flex_modules.module import Module
from networks.modules import ClassTokenLayer, PosEmbeddingLayer, LinearHead, LayerScale
import config.paths as paths


# Some of this code is from https://github.com/poojamangal15/Adaptive-Neural-Networks


def get_device() -> 'str':
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def make_str_filename_safe(s: str):
    prefix_char = 'x'
    forbidden_chars = [
        ('x', 'xx'),
        ('/', 'xa'),
        ('<', 'xb'),
        ('>', 'xc'),
        (':', 'xd'),
        ('"', 'xe'),
        ('/', 'xf'),
        ('\\', 'xg'),
        ('|', 'xh'),
        ('?', 'xi'),
        ('*', 'xj'),
        ('(', 'xk'),
        (')', 'xl'),
        ('.', 'xm'),
        (',', 'xn'),
        ('\'', 'xo')
    ]

    description = s
    description = description.replace(
        prefix_char, f"{prefix_char}{prefix_char}")
    for forbidden, replacement in forbidden_chars:
        description = description.replace(forbidden, replacement)
    return description


class SelfDescripting:
    def setv(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_description(self) -> str:
        res = f"{self.__class__.__name__}"
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                descr = val.get_description()
                res += f"_({descr})"
            except AttributeError:
                res += f"_{val}"
        return res

    def get_filename_safe_description(self) -> str:
        return make_str_filename_safe(self.get_description())

    def get_flat_dict(self) -> str:
        res = {}
        for name, val in self.__dict__.items():
            if name[:2] == "__":
                continue
            try:
                flatdict = val.get_flat_dict()
                for dname, dval in flatdict.items():
                    res[f"{name}.{dname}"] = dval
            except AttributeError:
                res[f"{name}"] = val
        return res


torch.serialization.add_safe_globals([SelfDescripting])


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> torch.Tensor:
    """
    Evaluates the model on the given dataloader and returns accuracy and F1 score.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    all_preds = []
    all_labels = []
    # Move model to the correct device and ensure correct data type
    model = model.to(device).to(torch.float32)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm.tqdm(dataloader):
            # Ensure images are on the same device and data type
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            outputs = model(images)  # Perform forward pass
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in the model.

    from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_in_mb(model: nn.Module) -> int:
    """
    Gets the models file size.

    adapted from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    f = io.BytesIO()
    torch.save(model.state_dict(), f)
    return len(f.getvalue())


def try_make_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_num_nodes():
    return int(os.environ.get("SLURM_NNODES", 1))


def load_dummy_data(
    num_classes: int = 1000,
    num_train: int = 1024,
    num_val: int = 512,
    num_test: int = 512,
    image_size: tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 512,
):
    """
    Generates dummy data loaders mimicking ImageNet.
    """

    # Helper to create random tensors
    def make_dataset(num_samples):
        images = torch.randn(num_samples, *image_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        return TensorDataset(images, labels)

    train_dataset = make_dataset(num_train)
    val_dataset = make_dataset(num_val)
    test_dataset = make_dataset(num_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dummy dataloaders created, BS:{batch_size}")
    return train_loader, val_loader, test_loader

IMAGENET_TRANSFORMS = [
    Resize(256),
    CenterCrop(224),
    RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BILINEAR),
    ColorJitter(0.4, 0.4, 0.4, 0.1),
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]

# ----- Mixup + CutMix -----
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,  # probability to switch between mixup and cutmix
    mode='batch',
    label_smoothing=0.11,
    num_classes=1000
)

mixup_fn_cifar100 = Mixup(
    mixup_alpha=0.8,  # Mixup parameter
    cutmix_alpha=1.0, # CutMix parameter
    cutmix_minmax=None,
    prob=1.0,         # apply either mixup or cutmix with 100% prob
    switch_prob=0.5,  # 50% chance to switch between mixup/cutmix
    mode='batch',
    label_smoothing=0.1,
    num_classes=100
)

def load_imagenet(data_dir=paths.IMAGENET_PATH, tmp_dir=paths.TMPDIR, batch_size=512):
    train_transform = Compose(IMAGENET_TRANSFORMS)
    test_transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    test_dataset = ImageFolder(data_dir / "val", transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=16)

    print(f"made dataloaders, BS:{batch_size}")
    return train_dataloader, val_dataloader, test_dataloader



def load_data(dataset, data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, resize=None, batch_size=64):
    """
    Loads data for CIFAR10 or CIFAR100

    Inspired by code from https://github.com/poojamangal15/Adaptive-Neural-Networks
    """
    normalizers = {
        CIFAR10: {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225], },
        CIFAR100: {'mean': [0.5070, 0.4865, 0.4409],
                   'std': [0.2673, 0.2564, 0.2761], }
    }

    train_transform = [
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    test_transform = [
        ToTensor(),
        Normalize(**normalizers[dataset])
    ]

    if resize:
        test_transform.insert(0, Resize(resize))
        train_transform.insert(0, Resize(resize))

    test_transform = Compose(test_transform)
    train_transform = Compose(train_transform)

    if tmp_dir:
        try_make_dir(data_dir)
        try_make_dir(tmp_dir)
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    train_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=True, download=True, transform=train_transform)
    test_dataset = dataset(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=False, download=True, transform=test_transform)
    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def load_wikitext(
    dataset_name: str = "wikitext-103-raw-v1",
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_dir: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Downloads (or loads from cache) a WikiText dataset, tokenises with the GPT-2
    tokeniser, packs sequences to max_seq_length with no padding, and returns
    (train_loader, val_loader, test_loader).

    Each batch is a tuple (input_ids, input_ids) of shape [B, max_seq_length].
    Labels are the same as inputs; the loss function shifts them by one position.

    Args:
        dataset_name: HuggingFace dataset config, e.g. "wikitext-103-raw-v1"
                      or "wikitext-2-raw-v1" for a smaller version.
        max_seq_length: Number of tokens per training example.
        batch_size: Examples per batch.
        num_workers: DataLoader worker processes.
        cache_dir: Optional path to override the default HuggingFace cache.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import Dataset as TorchDataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    raw = load_dataset("wikitext", dataset_name, cache_dir=cache_dir)

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    def pack(examples):
        # Flatten all token sequences in the batch into one list, then rechunk.
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // max_seq_length) * max_seq_length
        chunks = [ids[i: i + max_seq_length] for i in range(0, total, max_seq_length)]
        return {"input_ids": chunks}

    packed = tokenized.map(pack, batched=True, remove_columns=["attention_mask"])
    packed.set_format(type="torch", columns=["input_ids"])

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])  # [B, S]
        return ids, ids  # (input_ids, labels) — loss shifts internally

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            packed[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

    return make_loader("train", True), make_loader("validation", False), make_loader("test", False)


def load_openwebtext(
    max_seq_length: int = 1024,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_dir: str = None,
    val_size: int = 2000,
    test_size: int = 2000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # OpenWebText — GPT-2's original training data, better for LAMBADA preservation
    # than a Wikipedia-based corpus.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    raw = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir)["train"]

    # OpenWebText has only a train split — carve out val and test
    split = raw.train_test_split(test_size=val_size + test_size, seed=42)
    val_test = split["test"].train_test_split(test_size=test_size, seed=42)
    splits = {
        "train":      split["train"],
        "validation": val_test["train"],
        "test":       val_test["test"],
    }

    def tokenize(examples):
        return tokenizer(examples["text"])

    def pack(examples):
        ids = sum(examples["input_ids"], [])
        total = (len(ids) // max_seq_length) * max_seq_length
        chunks = [ids[i: i + max_seq_length] for i in range(0, total, max_seq_length)]
        return {"input_ids": chunks}

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        return ids, ids

    def make_loader(split_name: str, shuffle: bool) -> DataLoader:
        ds = splits[split_name]
        ds = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=num_workers)
        ds = ds.map(pack, batched=True, remove_columns=["attention_mask"], num_proc=num_workers)
        ds.set_format(type="torch", columns=["input_ids"])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )

    return make_loader("train", True), make_loader("validation", False), make_loader("test", False)


def flexible_model_copy(src: Union[nn.Module, dict[str, Any]], dest: nn.Module):
    if not isinstance(src, nn.Module):
        dest.load_state_dict(src)
        return

    if isinstance(src, Module):
        src.copy_to_base(dest)
        return

    if isinstance(dest, Module):
        dest.load_from_base(src)
        return

    dest.load_state_dict(src.state_dict())


def torch_serialize(obj, *args, **kwargs):
    with io.BytesIO() as f:
        torch.save(obj, f, *args, **kwargs)
        return f.getvalue()


def torch_deserialize(data: bytes, *args, **kwargs):
    with io.BytesIO(data) as f:
        return torch.load(f, *args, **kwargs)


def save_model(exp_name, model):
    exp_name = make_str_filename_safe(exp_name)
    with open(paths.TRAINED_MODELS / f"{exp_name}.pt", "wb") as f:
        torch.save(model.state_dict(), f)

def save_statedict(name, model):
    with open(paths.TRAINED_MODELS / f"{name}.pt", "wb") as f:
        torch.save(model.state_dict(), f)
    print("model state dict saved")


def load_model(exp_name, model_config):
    exp_name = make_str_filename_safe(exp_name)
    model = model_config.make_model()
    with open(paths.TRAINED_MODELS / f"{exp_name}.pt", "rb") as f:
        sdict = torch.load(f)
        model.load_state_dict(sdict)
    return model

@torch.no_grad()
def load_gpt2_weights_into_flexgpt(
        model,
        hf_model_name: str = "gpt2",
        cache_dir: str = None,
) -> None:
    """
    Loads HuggingFace GPT-2 pretrained weights into a max-level FlexGPT model.

    Notes for self:

    The max-level dimensions (hidden_dims[-1], num_heads[-1], mlp_dims[-1], num_layers)
    must match the target HF model exactly. GPT-2 Small ("gpt2") matches FlexGPTConfig
    with hidden_dims[-1]=768, num_heads[-1]=12, mlp_dims[-1]=3072, num_layers=12.

    GPT-2 uses Conv1D which stores weights as [in, out] — the transpose of nn.Linear's
    [out, in]. Every Conv1D weight is transposed before being written into FlexGPT.
    """
    from transformers import GPT2LMHeadModel

    cfg = model.config
    max_hidden = list(cfg.hidden_dims)[-1]
    max_heads  = list(cfg.num_heads)[-1]
    max_mlp    = list(cfg.mlp_dims)[-1]

    hf = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir=cache_dir)
    sd = hf.state_dict()
    del hf

    model.set_level_use(model.max_level())

    # Token embedding
    model.token_embedding.embedding.weight.data.copy_(sd["transformer.wte.weight"])

    # Positional embedding: GPT-2 [seq_len, hidden] → FlexGPT [1, seq_len, max_hidden]
    model.pos_embedding.embedding.data.copy_(sd["transformer.wpe.weight"].unsqueeze(0))

    for i in range(cfg.num_layers):
        block = model.blocks[i]
        pfx = f"transformer.h.{i}"

        # Pre-attention layer norm
        tmp_ln = nn.LayerNorm(max_hidden, eps=1e-5)
        tmp_ln.weight.data.copy_(sd[f"{pfx}.ln_1.weight"])
        tmp_ln.bias.data.copy_(sd[f"{pfx}.ln_1.bias"])
        block.ln_1.load_from_base(tmp_ln)

        # Self-attention — Conv1D weights need .T to become [out, in] (nn.Linear layout)
        tmp_mha = nn.MultiheadAttention(max_hidden, max_heads, batch_first=True)
        tmp_mha.in_proj_weight.data.copy_(sd[f"{pfx}.attn.c_attn.weight"].T)
        tmp_mha.in_proj_bias.data.copy_(sd[f"{pfx}.attn.c_attn.bias"])
        tmp_mha.out_proj.weight.data.copy_(sd[f"{pfx}.attn.c_proj.weight"].T)
        tmp_mha.out_proj.bias.data.copy_(sd[f"{pfx}.attn.c_proj.bias"])
        block.attn.load_from_base(tmp_mha)

        # Post-attention layer norm
        tmp_ln2 = nn.LayerNorm(max_hidden, eps=1e-5)
        tmp_ln2.weight.data.copy_(sd[f"{pfx}.ln_2.weight"])
        tmp_ln2.bias.data.copy_(sd[f"{pfx}.ln_2.bias"])
        block.ln_2.load_from_base(tmp_ln2)

        # MLP up-projection (c_fc): Conv1D [hidden, mlp] → .T → [mlp, hidden]
        tmp_fc = nn.Linear(max_hidden, max_mlp)
        tmp_fc.weight.data.copy_(sd[f"{pfx}.mlp.c_fc.weight"].T)
        tmp_fc.bias.data.copy_(sd[f"{pfx}.mlp.c_fc.bias"])
        block.mlp[0].load_from_base(tmp_fc)

        # MLP down-projection (c_proj): Conv1D [mlp, hidden] → .T → [hidden, mlp]
        tmp_proj = nn.Linear(max_mlp, max_hidden)
        tmp_proj.weight.data.copy_(sd[f"{pfx}.mlp.c_proj.weight"].T)
        tmp_proj.bias.data.copy_(sd[f"{pfx}.mlp.c_proj.bias"])
        block.mlp[3].load_from_base(tmp_proj)

    # Final layer norm
    tmp_ln_f = nn.LayerNorm(max_hidden, eps=1e-5)
    tmp_ln_f.weight.data.copy_(sd["transformer.ln_f.weight"])
    tmp_ln_f.bias.data.copy_(sd["transformer.ln_f.bias"])
    model.ln_f.load_from_base(tmp_ln_f)

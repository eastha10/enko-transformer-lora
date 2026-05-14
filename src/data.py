import pandas as pd
import torch
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.model.transformer import subsequent_mask

PAD = 0
UNK = 1
BOS = 2
EOS = 3

class TranslationDataset(Dataset):
    def __init__(self, df, src_sp, tgt_sp):
        self.df = df.reset_index(drop=True)
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.iloc[idx]["src"]
        tgt = self.df.iloc[idx]["tgt"]

        return encode_pair(
            src=src,
            tgt=tgt,
            src_sp=self.src_sp,
            tgt_sp=self.tgt_sp
        )


def encode_pair(src, tgt, src_sp, tgt_sp):
    src_ids = src_sp.encode(str(src), out_type=int) + [EOS]
    tgt_ids = [BOS] + tgt_sp.encode(str(tgt), out_type=int) + [EOS]

    return src_ids, tgt_ids

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = [
        torch.tensor(src_ids, dtype=torch.long)
        for src_ids in src_batch
    ]

    tgt_batch = [
        torch.tensor(tgt_ids, dtype=torch.long)
        for tgt_ids in tgt_batch
    ]

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD
    )

    return src_batch, tgt_batch

def make_src_mask(src_batch):
    src_mask = (src_batch != PAD).unsqueeze(-2)

    return src_mask

def make_tgt_mask(tgt_input):
    tgt_pad_mask = (tgt_input != PAD).unsqueeze(-2)

    size = tgt_input.size(1)
    future_mask = subsequent_mask(size).to(tgt_input.device)
    future_mask = future_mask.bool()

    tgt_mask = tgt_pad_mask & future_mask

    return tgt_mask

def make_batch(src, tgt):
    tgt_input = tgt[:, :-1]
    tgt_y = tgt[:, 1:]

    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt_input)

    return src, tgt_input, tgt_y, src_mask, tgt_mask

def build_dataloader(
    parquet_path,
    src_spm_path,
    tgt_spm_path,
    batch_size=32,
    shuffle=True,
    num_workers=0
):
    df = pd.read_parquet(parquet_path)

    required_columns = {"src", "tgt"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"parquet must contain columns: {required_columns}")
    df = df[["src", "tgt"]].dropna().reset_index(drop=True)
    src_sp = spm.SentencePieceProcessor()
    src_sp.load(src_spm_path)

    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(tgt_spm_path)

    dataset = TranslationDataset(
        df=df,
        src_sp=src_sp,
        tgt_sp=tgt_sp
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader
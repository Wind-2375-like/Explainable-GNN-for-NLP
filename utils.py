import time
import torch
import random
import pandas as pd
import numpy as np
from datetime import timedelta
from torch_geometric.data import Data
from collections import namedtuple
import warnings


class LoadData:
    def __init__(self, args):
        # print("prepare data")
        self.graph_path = "TextGCN_datasets/graph"
        self.args = args
        self.nodes = set()

        # node
        edges = []
        edge_weight = []
        with open(f"{self.graph_path}/{args.dataset}.txt", "r") as f:
            for line in f.readlines():
                val = line.split()
                if val[0] not in self.nodes:
                    self.nodes.add(val[0])
                if val[1] not in self.nodes:
                    self.nodes.add(val[1])
                edges.append([int(val[0]), int(val[1])])
                edge_weight.append(float(val[2]))

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weight)

        # feature
        self.nfeat_dim = len(self.nodes)
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = torch.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        # self.features = th.sparse.FloatTensor(indices, values, shape).to_dense()
        features = torch.sparse.FloatTensor(indices, values, shape)
        self.graph = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)

        # target
        target_fn = f"TextGCN_datasets/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # train val test split
        self.train_lst, self.test_lst = get_train_test(target_fn)


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)


def return_seed(nums=10):
    seed = random.sample(range(0, 100000), nums)
    return seed


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
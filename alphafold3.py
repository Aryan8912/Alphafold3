from __future__ import annotations

import random
import sh
from math import pi, sqrt
from pathlib import Path
from itertools import product, zip_longest
from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor, tensor, is_tensor
from torch.amap import autocast
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from torch.nn import (
    Module, 
    ModuleList,
    Linear,
    Sequential,
)

from beartype.typing import (
    Callable, 
    Dict,
    List,
    Literal,
    NamedTuple,
    Tuple,
)

from alphafold3_pytorch.tensor_typing import (
    Float, 
    Int,
    Bool,
    Shaped,
    typecheck,
    checkpoint,
    IS_DEBUGGING,
    DEEPSPEED_CHECKPOINTING
)

from alphafold3_pytorch.attention import (
    Attention,
    pad_at_dim,
    pad_or_slice_to,
    pad_to_multiple,
    concat_previous_window,
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed
)

from alphafold3_pytorch.inputs import (
    CONSTRAINT_DIMS,
    CONSTRAINTS,
    CONSTRAINTS_MASK_VALUE,
    IS_MOLECULE_TYPES,
    IS_NON_NA_INDICES,
    IS_PROTEIN_INDEX,
    IS_DNA_INDEX,
    IS_RNA_INDEX,
    IS_LIGAND_INDEX,
    IS_METAL_ION_INDEX,
    IS_BIOMOLECULE_INDICES,
    IS_NON_PROTEIN_INDICES,
    IS_PROTEIN,
    IS_DNA,
    IS_RNA,
    IS_LIGAND,
    IS_METAL_ION,
    MAX_DNA_NUCLEOTIDE_ID,
    MIN_RNA_NUCLEOTIDE_ID,
    MISSING_RNA_NUCLEOTIDE_ID,
    NUM_HUMAN_AMINO_ACIDS,
    NUM_MOLECULE_IDS,
    NUM_MSA_ONE_HOT,
    DEFAULT_NUM_MOLECULE_MODS,
    ADDITIONAL_MOLECULE_FEATS,
    hard_validate_atom_indices_ascending,
    BatchedAtomInput,
    Alphafold3Input,
    alphafold3_inputs_to_batched_atom_input,
)

from alphafold3_pytorch.common.biomolecule import (
    get_residue_constants,
)

from alphafold3_pytorch.nlm import (
    NLMEmbedding,
    NLMRegistry,
    remove_nlms
)

from alphafold3_pytorch.plm import (
    PLMEmbedding,
    PLMRegistry,
    remove_plms
)

from alphafold3_pytorch.model_utils import (
    ExpressCoordinatesInFrame,
    RigidFrom3Point,
    RigidFromReference3Points,
    calculate_weighted_rigid_align_weights,
    pack_one
)

from alphafold3_pytorch.utils.model_utils import (
    ExpressCoordinatesInFrame,
    RigidFrom3Point,
    RigidFromReference3Points,
    calculate_weighted_rigid_align_weights,
    pack_one
)
from alphafold3_pytorch.utils.utils import get_gpu_type, not_exists

from alphafold3_pytorch.utils.model_utils import distance_to_dgram

# personal libraries

from frame_averaging_pytorch import FrameAverage

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from colt5_attention import ConditionalRoutedAttention

from hyper_connections.hyper_connections_with_multi_input_stream  import HyperConnections

# other external libs

from tqdm import tqdm 
from loguru import logger

from importlib.metadata import version
from huggingface_hub import PytorchModelHubMixin, hf_hub_download

from bio.PDB.Structure import Structure
from Bio.PDB.Structure import StructureBuilder

# einstein notation related

import einx
from einops import rearrange, repear, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

"""
"""
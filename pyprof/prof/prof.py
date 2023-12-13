#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script reads the output (Python dictionary) created by parse.py.
For every kernel (line) in the input it determines
	module / class name e.g. torch.nn.functional
	operator name e.g. linear
	kernel parameters e.g. GEMM M, N, K, datatype
	bytes
	flops
	tensor core usage
	direction (fprop, bprop)
	and other things. Please see the tool usage.
"""

from .usage import parseArgs
from .output import Output
from .utility import Utility
from .pointwise import Pointwise
from .convert import Convert
from .blas import *
from .embedding import Embedding
from .reduction import *
from .dropout import Dropout
from .softmax import *
#from pooling import * # work in progress
from .linear import Linear
from .optim import Adam
from .misc import *
from .conv import Conv
from .activation import Activation
from .index_slice_join_mutate import Cat, Reshape, MaskedScatter, Gather, Nonzero, IndexSelect, MaskedSelect
from .recurrentCell import RNNCell
from .normalization import BatchNorm
from .randomSample import RandPerm
from .loss import MSELoss
from .data import Data
from .memory import OneZero, Fill, Full


def findFpropKernel(seq):
    #Find the last fprop kernel with the same seqId
    #First look at seqId and then at altSeqId
    for idx in reversed(range(len(kernels))):
        k = kernels[idx]
        if (seq in k['seqId']) and (k['dir'] == "fprop"):
            return idx

    for idx in reversed(range(len(kernels))):
        k = kernels[idx]
        if (seq in k['altSeqId']) and (k['dir'] == "fprop"):
            return idx

    return -1
    #print("Error: seqId {} not found.".format(seq), file=sys.stderr)
    #assert False


def foo(mod, op, d):
    op_0 = op[0].split("_dummy_")[0]
    if (op_0 == "linear"):
        xx = Linear(d)

    # rnncell, lstmcell, grucell
    elif (mod[0] in ["LSTMCell", "GRUCell"]) and (op_0 == "forward"):
        xx = RNNCell(d)

    elif op_0 in [
            "conv1d",
            "conv2d",
    ]:
        xx = Conv(d)

    elif (op_0 in Pointwise.ops):
    
        xx = Pointwise(d)

    elif (op_0 in Convert.ops):
        xx = Convert(d)

    elif op_0 in ["__matmul__", "matmul"]:
        xx = Matmul(d)

    elif op_0 == "embedding":
        xx = Embedding(d)

    #reduction
    elif op_0 == "sum":
        xx = Sum(d)

    elif op_0 == "mean":
        xx = Mean(d)

    elif op_0 == "norm":
        xx = Norm(d)

    elif op_0 == "dropout":
        xx = Dropout(d)

    #Index, Slice, Join, Mutate
    elif (op_0 == "cat"):
        xx = Cat(d)

    elif (op_0 == "reshape"):
        xx = Reshape(d)

    elif (op_0 == "masked_scatter_"):
        xx = MaskedScatter(d)

    elif (op_0 == "gather"):
        xx = Gather(d)

    elif (op_0 == "nonzero"):
        xx = Nonzero(d)

    elif (op_0 == "index_select"):
        xx = IndexSelect(d)

    elif (op_0 == "masked_select"):
        xx = MaskedSelect(d)

    #blas
    elif op_0 in ["addmm", "addmm_"]:
        xx = Addmm(d)

    elif op_0 == "mm":
        xx = Mm(d)

    elif op_0 == "bmm":
        xx = Bmm(d)

    #softmax
    elif op_0 == "softmax":
        xx = Softmax(d)

    elif op_0 == "log_softmax":
        xx = LogSoftmax(d)

    #loss
    elif op_0 == "mse_loss":
        xx = MSELoss(d)

    #optimizers
    elif op_0 == "adam":
        xx = Adam(d)

    #normalization
    elif op_0 == "batch_norm":
        xx = BatchNorm(d)

    #random
    elif op_0 == "randperm":
        xx = RandPerm(d)

    #memory
    elif op_0 in OneZero.ops:
        xx = OneZero(d)

    elif op_0 == "fill_":
        xx = Fill(d)

    elif op_0 == "full":
        xx = Full(d)

    #misc
    elif op_0 == "copy_":
        xx = Copy(d)

    elif op_0 == "clone":
        xx = Clone(d)

    elif op_0 == "contiguous":
        xx = Contiguous(d)

    elif op_0 == "any":
        xx = Any(d)

    elif (op_0 in Activation.ops):
        xx = Activation(d)

    elif op_0 == "to":
        xx = Convert(d)

    else:
        xx = Foo(d)

    return xx


def main():
    #Read cmd line arguments
    cmdArgs = parseArgs()

    output = Output(cmdArgs)
    output.header()

    idx = -1
    #Read in all the kernel info
    for line in cmdArgs.file:
        idx += 1
        kernel = eval(line)
        assert (kernel)
        kernels.append(kernel)

        k = kernel
        d = Data(k)

        mod = k['mod']
        op = k['op']

        flops = 0
        params = {"na": "na"}
        tc = "na"
        bytes = 0

        if (d.dir == "bprop"):
            d.seqMarker = k['seqMarker']
            seq = k['seqId']
            if len(seq) > 1:
                pass
            seq = k['seqId'][:1]
            assert (len(seq) == 1), seq
            #assert (seq[0] != 0)
            assert (len(d.seqMarker) > 0)
            #If there is no useful marker associated, use the
            #sequence number to find the kernel from fprop
            if len(d.argMarker) == 0:
                index = findFpropKernel(seq[0])
                if index >= 0:
                    d.argMarker = kernels[index]['marker']
                    d.modMarker = kernels[index]['reprMarkers']
                    mod = kernels[index]['mod']
                    op = kernels[index]['op']

                    d.layer = kernels[index]['layer']
                    d.trace = kernels[index]['trace']

        # Check if marker has our annotations
        if len(d.argMarker) and Utility.hasNVTX(d.argMarker[0]):

            xx = foo(mod, op, d)

            bytes = xx.bytes()
            flops = xx.flops()
            op = xx.op()
            params = xx.params()
            tc = xx.tc()

        if type(op) is list:
            if len(op):
                op = op[0]
            else:
                op = ""

        if type(mod) is list:
            if len(mod):
                mod = mod[0]
            else:
                mod = ""

        d.index = idx + 1

        # The following 8 come from operator class functions.
        d.setParams(params)
        d.tc = tc
        d.flops = flops
        d.bytes = bytes
        d.mod = mod
        d.op = op

        output.data(d)


kernels = []
if __name__ == '__main__':
    main()

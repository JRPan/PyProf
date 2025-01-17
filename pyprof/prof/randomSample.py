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

from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase


class RandPerm(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args

        assert (mod == "torch")
        assert (op.split("_dummy_")[0] == "randperm")
        assert (len(args) == 1)
        n = args[0]
        assert n['type'] == "int"
        self.n = n['value']

    def params(self):
        p = OrderedDict([('N', self.n)])
        return p

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        return self.n * Utility.typeToBytes("int64")

    def flops(self):
        # Depends on RNG but this is probably a reasonable assumption.
        return self.n * 3

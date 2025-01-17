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

#TODO: Add support for other optimizers.


class Adam(OperatorLayerBase):

    def __init__(self, d):
        marker = eval(d.argMarker[0])
        mod = marker['mod']
        op = marker['op']
        args = marker['args']

        self.marker = marker
        self.mod_ = mod
        self.op_ = op
        self.args = args
        self.sub = d.sub

        assert (op.split("_dummy_")[0] == "adam")
        assert (len(args) == 12) or (len(args) == 14)
        w, hw, m, v, g = args[0:5]
        assert (w['shape'] == m['shape'] == v['shape'] == g['shape'])
        assert (hw['shape'] == w['shape']) or (hw['shape'] == (0, ))  #hw could be null
        assert (w['type'] == m['type'] == v['type'] == g['type'] == hw['type'] == "tensor")
        assert (w['dtype'] == m['dtype'] == v['dtype'] == "float32")

        self.w = w
        self.g = g

    def params(self):
        p = OrderedDict([('T', self.w['shape']), ('wtype', self.w['dtype']), ('gtype', self.g['dtype'])])
        return p

    def flops(self):
        return 0

    def bytes(self):
        wshape = self.w['shape']
        wtype = self.w['dtype']
        gtype = self.g['dtype']
        b = 0

        elems = Utility.numElems(wshape)

        #Get time to stream read/write w, m, v
        b += 6 * elems * Utility.typeToBytes(wtype)

        #Get time to read "g"
        b += elems * Utility.typeToBytes(gtype)

        if wtype != gtype:  #mixed precision
            #Get time to write "hw
            b += elems * Utility.typeToBytes(gtype)

        return b if (self.sub == 0) else 0

    def tc(self):
        return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

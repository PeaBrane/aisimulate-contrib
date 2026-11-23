# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each request of an attention case owns whole KV pages.

SGLang's paged backends keep every ``page_size``-th token id of a request, divide it by
``page_size`` and read the request from offset zero of those pages (sglang v0.5.23
``flashattention_backend.py:1317-1330``). Serving's allocator starts every request on a page
boundary, so the collector's hand-built ``req_to_token`` has to as well: packed requests of
two tokens under 128-token pages all read the first request's slots."""

import ast
import math
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_attn.py"


class _Tensor(np.ndarray):
    def view(self, *shape):
        return self.reshape(*shape)

    def contiguous(self):
        return self


class _Pool:
    def __init__(self, size, max_context_len, **kwargs):
        self.req_to_token = np.zeros((size, max_context_len), dtype=np.int32)


def _allocator():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {"_request_stride", "create_req_to_token_pool"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    fake_torch = types.SimpleNamespace(
        arange=lambda *a, dtype=None, device=None: np.ndarray.view(np.arange(*a, dtype=dtype), _Tensor),
        int32=np.int32,
    )
    ns = {"torch": fake_torch, "ReqToTokenPool": _Pool, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def _read_locations(token_matrix, page_size):
    """The token slots a paged backend reads for each request: its strided page ids, expanded."""
    pages = token_matrix[:, ::page_size] // page_size
    offsets = np.arange(token_matrix.shape[1])
    return pages[:, offsets // page_size] * page_size + offsets % page_size


@pytest.mark.parametrize("page_size", [1, 64, 128])
@pytest.mark.parametrize(("batch_size", "total_len"), [(32, 2), (8, 129), (4, 128), (3, 1000), (1, 5)])
def test_a_paged_backend_reads_the_slots_each_request_wrote(page_size, batch_size, total_len):
    ns = _allocator()
    pool, written = ns["create_req_to_token_pool"](batch_size, total_len, page_size, "cpu", "cpu")
    written = np.asarray(written)
    assert np.array_equal(pool.req_to_token, written)
    assert np.array_equal(_read_locations(written, page_size), written)
    assert len(np.unique(written)) == batch_size * total_len
    # page 0 is the pool's reserved page, and the pool the collector sizes holds every slot
    stride = ns["_request_stride"](total_len, page_size)
    assert written.min() == page_size and written.max() < page_size + batch_size * stride
    assert stride % page_size == 0 and stride >= total_len


def test_unpaged_requests_stay_packed():
    """page_size 1 (triton, flashinfer, fa3 off SM100) keeps the layout the shipped rows were timed on."""
    ns = _allocator()
    _, written = ns["create_req_to_token_pool"](3, 4, 1, "cpu", "cpu")
    assert np.asarray(written).tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

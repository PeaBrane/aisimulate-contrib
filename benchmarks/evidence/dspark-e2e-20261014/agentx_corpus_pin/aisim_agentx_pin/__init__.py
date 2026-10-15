# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original dataset adapter; producer timing and request reconstruction are unchanged.

API source: SemiAnalysisAI/aiperf@754356e9a39acc6cc6afb242d123bb57c3fb6f75,
src/aiperf/dataset/loader/{base_hf_dataset,semianalysis_cc_traces_weka}.py.
Apache-2.0, NVIDIA Corporation & Affiliates. See the campaign COLLECTION.md.
"""

import hashlib
import json
import os
from pathlib import Path

from aiperf.dataset.loader.semianalysis_cc_traces_weka import (
    SemiAnalysisCCTracesWekaLoader,
)
from huggingface_hub import hf_hub_download

REPOSITORY = "semianalysisai/cc-traces-weka-062126"
REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
SIZE = 1847151435


class PinnedWekaLoader(SemiAnalysisCCTracesWekaLoader):
    hf_revision = REVISION

    def _load_hf_dataset(self):
        if self.hf_dataset_name != REPOSITORY:
            raise ValueError("campaign corpus loader cannot substitute another dataset")
        path = Path(hf_hub_download(REPOSITORY, "traces.jsonl", repo_type="dataset", revision=REVISION))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != SIZE or digest != SHA256:
            raise ValueError("AgentX corpus content differs from the immutable reference")
        dataset = super()._load_hf_dataset()
        if len(dataset) != 393:
            raise ValueError("pinned AgentX corpus must contain all 393 source traces")
        receipt = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "file": "traces.jsonl",
            "sha256": digest,
            "bytes": SIZE,
            "rows": len(dataset),
            "local_content_verified": True,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        }
        destination = Path(os.environ["DSPARK_AGENTX_CORPUS_RECEIPT"])
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(destination)
        return dataset

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with the pinned producer environment; no corpus or network needed."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aisim_agentx_pin as pin


class LoaderTests(unittest.TestCase):
    def test_content_receipt_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus"
            path.write_bytes(b"verified fixture")
            receipt = Path(directory) / "receipt.json"
            loader = object.__new__(pin.PinnedWekaLoader)
            loader.hf_dataset_name = pin.REPOSITORY
            with (
                patch.object(pin, "hf_hub_download", return_value=str(path)) as download,
                patch.object(pin, "SIZE", path.stat().st_size),
                patch.object(pin, "SHA256", hashlib.sha256(path.read_bytes()).hexdigest()),
                patch.object(pin.SemiAnalysisCCTracesWekaLoader, "_load_hf_dataset", return_value=list(range(393))),
                patch.dict(pin.os.environ, {"DSPARK_AGENTX_CORPUS_RECEIPT": str(receipt)}),
            ):
                self.assertEqual(len(loader._load_hf_dataset()), 393)
                self.assertEqual(download.call_args.kwargs["revision"], pin.REVISION)
                self.assertTrue(json.loads(receipt.read_text())["local_content_verified"])
                self.assertFalse(receipt.with_suffix(".tmp").exists())
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "content differs"):
                    loader._load_hf_dataset()


if __name__ == "__main__":
    unittest.main()

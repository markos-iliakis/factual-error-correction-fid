#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
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
#
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    references = []
    predictions = []

    parser = ArgumentParser()
    parser.add_argument("prediction_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    with open(args.prediction_file) as inputs, open(args.out_file, "w+") as final:

        for line_in in inputs:
            instance = json.loads(line_in)

            final.write(
                json.dumps(
                    {
                        "master_explanation": instance["masked_inds"],
                        "original_claim": instance["sentence1"],
                        "original_evidence": instance["sentence2"],
                        "target": instance["target"],
                        "original_id": instance["original"]["original"]["original_id"],
                        "claim_id": instance["original"]["original"]["claim_id"],
                        "actual_label": instance["original"]["veracity"],
                    }
                )
                + "\n"
            )
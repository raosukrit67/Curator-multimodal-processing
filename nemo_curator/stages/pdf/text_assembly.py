# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Stage for assembling final structured output from all extraction modalities.

Combines text extracted by Nemotron Parse with visual descriptions from
Nemotron Nano VL into a unified per-page JSON structure.
"""

import json

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class TextAssemblyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Assemble final structured output from extracted content.

    Merges text regions (from Parse) and visual analyses (from VL model)
    into a unified per-page structure, organized by content type.

    Output format per document:
    {
        "pages": [
            {
                "page_number": 0,
                "text_blocks": [{"class_name": "Text", "bbox": [...], "text": "..."}],
                "tables": [{"bbox": [...], "latex": "...", "description": "..."}],
                "figures": [{"bbox": [...], "class_name": "Picture", "description": "..."}],
                "full_text": "concatenated text in reading order"
            }
        ]
    }

    Args:
        routed_content_field: Column with routed content.
        analysis_results_field: Column with VL analysis results.
        output_field: Column for storing assembled output.
    """

    def __init__(
        self,
        routed_content_field: str = "routed_content",
        analysis_results_field: str = "analysis_results",
        output_field: str = "assembled_content",
    ):
        self.routed_content_field = routed_content_field
        self.analysis_results_field = analysis_results_field
        self.output_field = output_field

        self.name = "text_assembly"
        self.resources = Resources(cpus=0.5)
        self.batch_size = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.routed_content_field, self.analysis_results_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.routed_content_field,
            self.analysis_results_field,
            self.output_field,
        ]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        assembled_list = []

        for routed_json, analysis_json in zip(
            df[self.routed_content_field],
            df[self.analysis_results_field],
            strict=True,
        ):
            try:
                routed_pages = json.loads(routed_json)
                analyses = json.loads(analysis_json)

                if not routed_pages:
                    assembled_list.append(json.dumps({"pages": []}))
                    continue

                # Index VL analyses by (page_number, bbox) for lookup
                analysis_map = {}
                for a in analyses:
                    key = (a["page_number"], tuple(a["bbox"]))
                    analysis_map[key] = a.get("description", "")

                pages = []
                for page_data in routed_pages:
                    page_num = page_data["page_number"]
                    text_blocks = []
                    tables = []
                    figures = []
                    all_text_parts = []

                    # Process text regions (already extracted by Parse)
                    for region in page_data.get("text_regions", []):
                        cls = region["class_name"]
                        text = region.get("text", "").strip()

                        if cls == "Table":
                            table_entry = {
                                "bbox": region["bbox"],
                                "latex": text,
                            }
                            # Check if VL analysis exists for this table
                            key = (page_num, tuple(region["bbox"]))
                            if key in analysis_map:
                                table_entry["description"] = analysis_map[key]
                            tables.append(table_entry)
                            if text:
                                all_text_parts.append(f"[Table: {text}]")
                        else:
                            text_blocks.append({
                                "class_name": cls,
                                "bbox": region["bbox"],
                                "text": text,
                            })
                            if text:
                                all_text_parts.append(text)

                    # Process VL regions (visual content with descriptions)
                    for region in page_data.get("vl_regions", []):
                        cls = region["class_name"]
                        bbox = region["bbox"]
                        key = (page_num, tuple(bbox))
                        description = analysis_map.get(key, "")

                        if cls == "Table":
                            tables.append({
                                "bbox": bbox,
                                "latex": region.get("text", ""),
                                "description": description,
                            })
                            if description:
                                all_text_parts.append(
                                    f"[Table description: {description}]"
                                )
                        else:
                            figures.append({
                                "bbox": bbox,
                                "class_name": cls,
                                "description": description,
                            })
                            if description:
                                all_text_parts.append(
                                    f"[{cls}: {description}]"
                                )

                    pages.append({
                        "page_number": page_num,
                        "text_blocks": text_blocks,
                        "tables": tables,
                        "figures": figures,
                        "full_text": "\n\n".join(all_text_parts),
                    })

                assembled_list.append(json.dumps({"pages": pages}))

            except Exception as e:
                logger.error(f"Text assembly failed: {e}")
                assembled_list.append(json.dumps({"pages": []}))

        df[self.output_field] = assembled_list

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

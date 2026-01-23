#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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


# limitations under the License.
#

from nomad.datamodel.metainfo.basesections.v2 import System as SystemV2
from nomad.normalizing import Normalizer as NomadNormalizer


class Normalizer(NomadNormalizer):
    """
    Extended base class for normalizer plugins with additional functionality.

    Inherits from nomad.normalizing.Normalizer to ensure compatibility with
    NOMAD's plugin system.
    """

    def _representative_system(self, archive) -> 'SystemV2 | None':
        """Select a representative system for this entry.

        For v2 schema (archive.data with model_system):
        - Uses archive.data.representative_system_index if available
        - Falls back to the last model_system if index is -1 or not set
        - Returns the system marked as is_representative if found

        Args:
            archive: EntryArchive containing the data to normalize

        Returns:
            The representative ModelSystem or None if no systems exist
        """
        result = None

        # Try to find workflow information and select the representative system
        # based on it
        workflow = archive.workflow2

        if workflow:
            try:
                iscc = workflow.results.calculation_result_ref
                system = iscc.system_ref
                if system is not None:
                    result = iscc
            except Exception:
                pass

        if result is None:
            # Check if archive.data exists and has non-empty model_system
            if (
                not archive.data
                or not hasattr(archive.data, 'model_system')
                or len(archive.data.model_system) == 0
            ):
                self.logger.warning('no model_system found in archive.data')
                return None

            model_systems = archive.data.model_system

            # Try to use representative_system_index if it exists
            if hasattr(archive.data, 'representative_system_index'):
                rep_idx = archive.data.representative_system_index
                # Handle negative index (e.g., -1 for last element)
                if rep_idx is not None and (
                    -len(model_systems) <= rep_idx < len(model_systems)
                ):
                    result = model_systems[rep_idx]

        # Fallback: find system with is_representative flag
        if result is None:
            for system in model_systems:
                if hasattr(system, 'is_representative') and system.is_representative:
                    result = system
                    break

        # Final fallback: use last system in list
        if result is None:
            result = model_systems[-1]
            self.logger.info(
                'no explicit representative system found, using last model_system'
            )

        return result

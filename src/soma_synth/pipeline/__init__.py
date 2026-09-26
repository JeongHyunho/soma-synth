"""The generation pipeline: canonical stages, run identity, run directory and provenance.

Implements ``docs/guides/GENERATION_PIPELINE_STANDARD.md``, which is in turn the execution spec
for the sealed ``PIPELINE_GOVERNANCE.md`` §5.1, §10.1 and §10.2.

This package does not generate anything. It names the stages, gives a run an identity, writes
the record §10.1 requires and collects the provenance §10.2 requires, around generators that
still own their own execution.
"""

from soma_synth.pipeline.stages import (  # noqa: F401
    GOVERNANCE_STAGES,
    SOURCE_VARIANT_STAGE,
    Corpus,
    GenerationPolicy,
    PipelineError,
    PipelineRegistry,
    PostStep,
    SourcePipeline,
    Stage,
    default_registry,
    load,
)

__all__ = [
    "GOVERNANCE_STAGES",
    "SOURCE_VARIANT_STAGE",
    "Corpus",
    "GenerationPolicy",
    "PipelineError",
    "PipelineRegistry",
    "PostStep",
    "SourcePipeline",
    "Stage",
    "default_registry",
    "load",
]

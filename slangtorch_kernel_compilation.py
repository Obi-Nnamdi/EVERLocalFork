import slangtorch


from pathlib import Path

# BRDF Eval Kernels

# Constants for Kernel Compilation
MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL = 400  # How many incoming light directions are we using at maximum? (used for checkpointing incoming light kernel)
USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS = False
MAX_NUMEL_FOR_SLANGTORCH = 4294967295 // 2  # close to INT32 overflow

brdf_eval_kernels = slangtorch.loadModule(
    str(Path(__file__).parent / "ever/splinetracers/slang/brdf_eval.slang"),
    defines={
        "MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL": MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL,
        "USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS": int(
            USE_CHECKPOINTING_FOR_INCOMING_LIGHT_PROBE_BACKWARD_PASS
        ),
    },
    skipNinjaCheck=True,
)
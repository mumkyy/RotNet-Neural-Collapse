#!/bin/bash
# Submit ALL force-collapse STL jigsaw9 downstream probes (4 variants x 17), each its own job. No dependencies.
# Backbone checkpoints read from SCRATCH via --pretrained_path.
set -e
cd "$(dirname "$0")"
SCRATCH=/scratch/amr239/ai342/RNC_experiments/
for f in $(find config/STL10/jigsaw9/MSE/Collapsed/NC1Layerwise \
                config/STL10/jigsaw9/MSE/Collapsed/NC1Final \
                config/STL10/jigsaw9/MSE/Collapsed/NC3Layerwise \
                config/STL10/jigsaw9/MSE/Collapsed/NC3Final -name '*.py' | sort); do
  exp=${f#config/}; exp=${exp%.py}
  jid=$(sbatch --parsable --job-name="$(basename "$exp")" \
        submit_one.sh --exp "$exp" --output_model_path "$SCRATCH" --pretrained_path "$SCRATCH" --num_workers 8 --cuda)
  echo "$jid  $exp"
done

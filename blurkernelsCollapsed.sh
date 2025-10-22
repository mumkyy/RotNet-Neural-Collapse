#!/bin/bash

BASE="CIFAR10/gaussian/blur/"
CONFIG="config/CIFAR10/gaussian/blur/backbone/"

for file in ${CONFIG}*.py; do 
    fileBase=$(basename "$file" | sed s/.py// )
    BACK="${BASE}backbone/${fileBase}"
    KS_TAG=$(echo "$fileBase" | sed 's/CIFAR10_Gaussian_Blur_NIN4blocks_Collapsed_MSE_//')
    DOWN4="${BASE}conv4/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv4_feats_Collapsed_${KS_TAG}"
    DOWN2="${BASE}conv2/CIFAR10_ConvClassifier_Gaussian_Blur_NIN4blocks_Conv2_feats_Collapsed_${KS_TAG}"

    # upstream
    jid_up=$(sbatch --parsable --job-name="gpu_collapsed_gb_kernel_up_${KS_TAG}" --export=ALL,EXP_NAME="$BACK",KS_TAG="$KS_TAG" ./submit.sh)
    echo "Submitted upstream: $jid_up"

    # conv2 waits for upstream to succeed
    jid_c2=$(sbatch --parsable --dependency=afterok:${jid_up} --job-name="gpu_collapsed_gb_kernel_conv2_${KS_TAG}" --export=ALL,EXP_NAME="$DOWN2",KS_TAG="$KS_TAG" ./submit.sh)
    echo "Submitted conv2: $jid_c2 (after $jid_up)"

    # conv4 waits for upstream to succeed
    jid_c4=$(sbatch --parsable --dependency=afterok:${jid_up} --job-name="gpu_collapsed_gb_kernel_conv4_${KS_TAG}" --export=ALL,EXP_NAME="$DOWN4",KS_TAG="$KS_TAG" ./submit.sh)
    echo "Submitted conv4: $jid_c4 (after $jid_up)"
    

done


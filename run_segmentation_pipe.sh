#!/bin/bash


usage() {
    echo "Usage: $0 [--step1] [--step2] [--all]"
    echo " --step1: Render multiview images of 3D models"
    echo " --step2: Run CamHMR to get unaligned SMPL and front view"
    echo " --step3: Get multiview masks using front frame from CamHMR and tracking"
    echo " --step4: Reproject the depth using the masks"
    echo " --step5: Align the SMPL with clean person pointcloud"
    echo "  --all: Run all steps"
    exit 1
}

## Add more steps as needed
# Initialize flags
RUN_STEP1=false
RUN_STEP2=false
RUN_STEP3=false
RUN_STEP4=false
RUN_STEP5=false

#  Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, run first 3 steps by default
    # Step4 is optional and only used for proper runtime calculation and can be run with --step4
    RUN_STEP1=true
    RUN_STEP2=true
    RUN_STEP3=true 
    RUN_STEP4=true
    RUN_STEP5=true
else
    while [[ $# -gt 0 ]]; do
        case $1 in
            --step1)
                RUN_STEP1=true
                shift
                ;;
            --step2)
                RUN_STEP2=true
                shift
                ;;
            --step3)
                RUN_STEP3=true
                shift
                ;;
            --step4)
                RUN_STEP4=true
                shift
                ;;
            --step5)
                RUN_STEP5=true
                shift
                ;;
            --all)
                RUN_STEP1=true
                RUN_STEP2=true
                RUN_STEP3=true
                RUN_STEP4=true
                RUN_STEP5=true
                RUN_STEP6=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option $1"
                usage
                ;;
        esac
    done
fi

# Activate the conda environment for image inpainting
source ~/.bashrc
PROJECT_ROOT="$(pwd)"
RUN_ID="test_run"  # Change this to your desired run ID
mkdir -p output/${RUN_ID}

if [ "$RUN_STEP1" = true ]; then
    echo "Running Step 1: Render multiview images of 3D models"

    conda activate HY21_ENV
    cd ${PROJECT_ROOT}/scripts

    python render_scripts/render_for_clip.py --input_dir ${PROJECT_ROOT}/output/${RUN_ID}/combined_3D \
                        --output_dir RENDERED_combined_3D

    # move the RENDERED_combined_3D to DISTI_OUTPUTS
    mv RENDERED_combined_3D ${PROJECT_ROOT}/output/${RUN_ID}/

    conda deactivate
fi

if [ "$RUN_STEP2" = true ]; then
    echo "Running Step 2: Run CamHMR to get unaligned SMPL and front view"

    conda activate CAMHMR_ENV
    cd ${PROJECT_ROOT}/external/CameraHMR

    python demo_hoi3dgen.py --input_folder ${PROJECT_ROOT}/output/${RUN_ID}/RENDERED_combined_3D \
                    --output_folder ${PROJECT_ROOT}/output/${RUN_ID}/camHMR_output  #--save_partial_mesh True

    conda deactivate
fi

if [ "$RUN_STEP3" = true ]; then
    echo "Running Step 3: Get multiview masks using front frame from CamHMR and tracking"

    conda activate SANA_ENV
    cd ${PROJECT_ROOT}/external/Grounded-SAM-2/

    python find_best_frame_using_camHMR.py --camHMR-dir ${PROJECT_ROOT}/output/${RUN_ID}/camHMR_output \
                    --text-prompt-csv ${PROJECT_ROOT}/assets/${RUN_ID}/prompts_deconstructed.csv \
                    --output-dir whitebg_${RUN_ID}

    python tracking_for_disti.py --base_dir whitebg_${RUN_ID} \
            --original_dir ${PROJECT_ROOT}/output/${RUN_ID}/RENDERED_combined_3D \
            --text-prompt-csv ${PROJECT_ROOT}/assets/${RUN_ID}/prompts_deconstructed.csv

    conda deactivate
fi

if [ "$RUN_STEP4" = true ]; then
    echo "Running Step 4: Reproject the depth using the masks"
    conda activate CAMHMR_ENV

    cd ${PROJECT_ROOT}/scripts/segmentMeshes/

    python reproject_mv_human.py --parent_dir ${PROJECT_ROOT}/output/${RUN_ID}/RENDERED_combined_3D \
                        --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/SEGMENTATION_OUTPUTS \
                        --segmentation_subject person

    python reproject_mv_human.py --parent_dir ${PROJECT_ROOT}/output/${RUN_ID}/RENDERED_combined_3D \
                        --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/SEGMENTATION_OUTPUTS \
                        --segmentation_subject object

    # Clean the pointclouds
    python remove_pc_intersections.py --base_dir ${PROJECT_ROOT}/output/${RUN_ID}/SEGMENTATION_OUTPUTS

    conda deactivate
fi

if [ "$RUN_STEP5" = true ]; then
    echo "Running Step 5: Align the SMPL with clean person pointcloud"
    conda activate CAMHMR_ENV
    cd ${PROJECT_ROOT}/scripts/segmentMeshes/

    # for baseline we use reproject_and_align_camHMR_partial.py for other cases we use reproject_and_align_camHMR.py
    python reproject_and_align_camHMR.py --input_mesh_dir ${PROJECT_ROOT}/output/${RUN_ID}/camHMR_output \
                        --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/SEGMENTATION_OUTPUTS
    conda deactivate
fi
#!/bin/bash


usage() {
    echo "Usage: $0 [--step1] [--step2] [--all]"
    echo "  --step1: Run Sana Model Inference"
    echo "  --step2: Run Depth Anything Inference"
    echo " --step3: Run FLUX retexturing"
    echo " --step4: Mask to remove background"
    echo " --step5: Lift to 3D mesh"
    echo " --step6: Texture the 3D mesh"
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
RUN_STEP6=false

#  Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, run first 3 steps by default
    # Step4 is optional and only used for proper runtime calculation and can be run with --step4
    RUN_STEP1=true
    RUN_STEP2=true
    RUN_STEP3=true 
    RUN_STEP4=true
    RUN_STEP5=true
    RUN_STEP6=true
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
            --step6)
                RUN_STEP6=true
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
    echo "Running Step 1: Sana Model Inference"
    conda activate SANA_ENV

    # create prompts.txt from prompts_deconstructed.csv
    cd ${PROJECT_ROOT}/scripts/utils/
    python create_txt_from_csv.py --input_csv ${PROJECT_ROOT}/assets/${RUN_ID}/prompts_deconstructed.csv \
        --output_txt ${PROJECT_ROOT}/assets/${RUN_ID}/prompts.txt


    cd ${PROJECT_ROOT}/external/Sana

    python scripts/inference.py --config configs/sana1-5_config/1024ms/Sana_1600M_1024px_infer.yaml  \
        --txt_file ${PROJECT_ROOT}/assets/${RUN_ID}/prompts.txt \
        --model_path ${PROJECT_ROOT}/weights/epoch_1050_step_25201.pth \
        --step 50 --is_bash_trigger True --bash_suffix ${RUN_ID} --add_camera_control True --is_baseline_run False

    mv ${PROJECT_ROOT}/vis/bash_image_outputs_${RUN_ID} ${PROJECT_ROOT}/output/${RUN_ID}/SANA_OUTPUTS/
    rm -rf ${PROJECT_ROOT}/vis

    conda deactivate
fi

if [ "$RUN_STEP2" = true ]; then
    echo "Running Step 2: Depth Anything Inference"
    conda activate SANA_ENV
    # create prompts.txt from prompts_deconstructed.csv
    cd ${PROJECT_ROOT}/scripts/flux/

    python create_depth_images.py --input_folder ${PROJECT_ROOT}/output/${RUN_ID}/SANA_OUTPUTS/ \
        --output_folder ${PROJECT_ROOT}/output/${RUN_ID}/DEPTH_OUTPUTS/
    
    conda deactivate
fi


if [ "$RUN_STEP3" = true ]; then
    echo "Running Step 3: FLUX Retexturing"
    conda activate SANA_ENV

    # create prompts.txt from prompts_deconstructed.csv
    cd ${PROJECT_ROOT}/scripts/flux/

    python retexture_depth_images.py --depth_dir ${PROJECT_ROOT}/output/${RUN_ID}/DEPTH_OUTPUTS/ \
        --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/retex_out/ \
        --prompt_file ${PROJECT_ROOT}/assets/${RUN_ID}/prompts.txt \
        --num_views 3

    conda deactivate
fi

if [ "$RUN_STEP4" = true ]; then
    echo "Running Step 4: Mask and lift objects into 3D"
    conda activate SANA_ENV

     cd ${PROJECT_ROOT}/external/Grounded-SAM-2/

    # Run for combined Human + Object images
    # for baseline we use SANA_OUTPUTS, otherwise use retex_out
    python get_masks_and_make_bg_white.py --parent-dir ${PROJECT_ROOT}/output/${RUN_ID}/retex_out \
                                --output-dir ${PROJECT_ROOT}/output/${RUN_ID}/white_bg_combined \
                                --text-prompt-csv ${PROJECT_ROOT}/assets/${RUN_ID}/prompts_deconstructed.csv --zoom_in True

    conda deactivate
fi

if [ "$RUN_STEP5" = true ]; then

    echo "Running Step 5: Lift to 3D"

    echo "Now Generating Meshes"
    conda activate SANA_ENV

    cd ${PROJECT_ROOT}/external/HunYuan3D-2/

    python HY2_mesh.py --input_dir ${PROJECT_ROOT}/output/${RUN_ID}/white_bg_combined \
                        --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/intermidiate_mesh \
                        --max_faces 20000

    conda deactivate

fi

if [ "$RUN_STEP6" = true ]; then

    echo "Running Step 6: Texturing the 3D mesh"

    echo "Now Generating Meshes"
    conda activate HY21_ENV

    cd ${PROJECT_ROOT}/external/Hunyuan3D-2.1/
    export HY3DGEN_MODELS=${HF_HOME}

    python HY21_only_tex.py --input_mesh_dir ${PROJECT_ROOT}/output/${RUN_ID}/intermidiate_mesh \
                            --input_dir ${PROJECT_ROOT}/output/${RUN_ID}/white_bg_combined \
                            --output_dir ${PROJECT_ROOT}/output/${RUN_ID}/combined_3D --extension png

    conda deactivate
fi
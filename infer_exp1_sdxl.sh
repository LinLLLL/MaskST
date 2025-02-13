for content_prompt in  "dog"  "truck"  "horse"  "automobile" "frog" "airplane" "ship" "bird"  "cat" "deer" # 
do
for ob in "human" "laptop" "flower" "bench" "rocket" 
do
CUDA_VISIBLE_DEVICES=0 python infer_v1_sdxl.py \
        --less_condition  \
        --target_content $ob \
        --image_root_path "data/object_test" \
        --content_prompt $content_prompt \
        --theta 0.8
done
done
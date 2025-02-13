gpu=$1
scale=$2
neg_content_scale=$3
method=$4
less_condition=$5


if [[ -n $less_condition ]]; then
echo "Evaluating the Masking Model......"
for content_prompt in  "frog" "airplane" "automobile" "bird" "cat" "deer" "dog" "horse" "ship" "truck"
do
for ob in "human" "laptop" "flower" "bench" "rocket"
do
    CUDA_VISIBLE_DEVICES=$gpu python infer_v1_sd15.py \
    --less_condition  \
    --target_content $ob \
    --image_root_path "data/object_test" \
    --content_prompt $content_prompt \
    --method ours \
    --scale $scale
done
done
else
echo "Evaluating the baseline model......"
for content_prompt in  "frog" "airplane" "automobile" "bird" "cat" "deer" "dog" "horse" "ship" "truck"
do
for ob in "human" "laptop" "flower" "bench" "rocket"
do
    CUDA_VISIBLE_DEVICES=$gpu python infer_v1_sd15.py \
    --target_content $ob \
    --content_prompt $content_prompt \
    --image_root_path "data/object_test" \
    --neg_content_scale $neg_content_scale \
    --method $method \
    --scale $scale
done
done
fi
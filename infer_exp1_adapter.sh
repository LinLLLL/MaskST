gpu=$1
less_condition=$2

if [[ -n $less_condition ]]; then
for ob in "frog" "airplane" "automobile" "bird" "cat" "deer" "dog" "horse" "ship" "truck"
do
for epoch in 400 800 1200 1600 2000 2800 4000
do
        CUDA_VISIBLE_DEVICES=$gpu python infer_sd15_adapter.py \
        --less_condition  \
        --target_content $ob \
        --image_root_path "data/object_test" \
        --load_epoch $epoch 
done
done
else
for ob in "frog" "airplane" "automobile" "bird" "cat" "deer" "dog" "horse" "ship" "truck"
do
for epoch in 400 800 1200 1600 2000 2800 4000
do
        CUDA_VISIBLE_DEVICES=$gpu python infer_sd15_adapter.py \
        --target_content $ob \
        --image_root_path "data/object_test" \
        --load_epoch $epoch 
done
done
fi
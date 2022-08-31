python3 tensorrt_optimize.py --data_dir datasets_opencv/ \
	--preprocess crop \
	--preprocess extreme \
	--data_augs 1 \
	--img_shape "200,66,3" \
	--batch_size 32 \
	--model_path trained_models/pilotnet.h5 \
	--model_name pilotnet \
	--eval_base True \
 	--precision all
    #  \
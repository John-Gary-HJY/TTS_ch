CUDA_VISIBLE_DEVICES=2 uv run python testing.py --model_dir ./experiments/acoustic_model_demo --dp_model_dir ./experiments/duration_predictor_demo --restore_file pretrained_acoustic.pt --dp_restore_file pretrained_duration_predictor.pth.tar --fpath test_script/test_demo.txt --speaker cn17f


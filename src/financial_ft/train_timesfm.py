import timesfm, torch, os

# force Torch to stay on CPU even if CUDA drivers are present
os.environ["CUDA_VISIBLE_DEVICES"] = ""

hp   = timesfm.TimesFmHparams(backend="cpu",
                              context_len=512,
                              horizon_len=128,
                              per_core_batch_size=4)   # a few samples fit in RAM
ckpt = timesfm.TimesFmCheckpoint(
           huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
tfm  = timesfm.TimesFm(hparams=hp, checkpoint=ckpt)     #  ✅ imports on CPU
print("Parameters (M):", tfm.model.num_parameters()/1e6)

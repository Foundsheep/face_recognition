import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.model_loader import Classfier
from image_processing.data_util import load_data
import argparse
import datetime
from pathlib import Path

from configs import Config

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # callbacks 
    early_stop_callback = EarlyStopping(monitor="valid_loss",
                                        patience=4,
                                        mode="min",
                                        verbose=True)
    
    train_dl = load_data(root_folder=args.root_folder,
                         shuffle=args.shuffle,
                         batch_size=args.batch_size,
                         num_workers=args.dl_num_workers,
                         read_step=args.read_step,
                         is_train=True)

    val_dl = load_data(root_folder=args.root_folder,
                       shuffle=False,
                       batch_size=args.batch_size,
                       num_workers=args.dl_num_workers,
                       read_step=args.read_step * 3,
                       is_train=False)

    test_dl = load_data(root_folder=args.root_folder,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.dl_num_workers,
                        read_step=args.read_step * 4,
                        is_train=False)


    trainer = L.Trainer(accelerator="gpu" if Config.DEVICE == "cuda" else Config.DEVICE,
                        min_epochs=args.min_epochs,
                        max_epochs=args.max_epochs,
                        log_every_n_steps=args.log_every_n_steps,
                        default_root_dir=str(Config.TRAINING_LOG_DIR / f"{timestamp}_{args.model_name}_epochs{args.max_epochs}_batch{args.batch_size}"),
                        callbacks=[early_stop_callback] if args.use_early_stop == "Y" else [])
    
    model = Classfier()
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl)
    
    if test_dl is not None:
        trainer.test(model, test_dl)


if __name__ == "__main__":
    
    current_dir = Path(__file__).absolute().parent
    saved_image_dir = current_dir / Path("image_processing") / "image_saved"

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default=str(saved_image_dir))
    parser.add_argument("--model_name", type=str, default=Config.CLASSIFICATION_MODEL_NAME)
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS)
    parser.add_argument("--min_epochs", type=int, default=Config.MIN_EPOCHS)
    parser.add_argument("--shuffle", type=bool, default=Config.SHUFFLE)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--dl_num_workers", type=int, default=Config.DL_NUM_WORKERS)
    parser.add_argument("--log_every_n_steps", type=int, default=Config.LOG_EVERY_N_STEPS)
    parser.add_argument("--read_step", type=int, default=Config.READ_STEP)
    parser.add_argument("--use_early_stop", type=bool, default=Config.USE_EARLY_STOP)


    args = parser.parse_args()
    main(args)

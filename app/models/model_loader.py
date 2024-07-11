import timm
import torch
from torch import nn, optim
import lightning as L
import time
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent))

from configs import Config


def count_parameters(model, for_all=False):
    value = None
    if for_all:
        value = sum(p.numel() for p in model.parameters())
    else:
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    return value


# def get_model(model_name, num_classes, is_freeze=True):
#     model = timm.create_model(model_name, pretrained=True)
#     layers = list(model.children())
#     in_features = layers[-1].in_features
#     linear = nn.Linear(in_features=in_features,
#                        out_features=num_classes)
    
#     model_upper_parts = nn.Sequential(*layers[:-1])
#     if is_freeze:
#         for param in model_upper_parts.parameters():
#             param.requires_grad = False
    
#     model = nn.Sequential(model_upper_parts, linear)
#     return model

class Classfier(L.LightningModule):
    def __init__(self, model_name=Config.CLASSIFICATION_MODEL_NAME, num_classes=Config.CLASSIFICATION_NUM_CLASSES):
        super().__init__()
        self.model = self._get_model(model_name=model_name,
                                     num_classes=num_classes)
        self.num_classes = num_classes
        self.model_name = model_name
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def _get_model(self, model_name, num_classes, is_freeze=True):
        model = timm.create_model(model_name, pretrained=True)
        layers = list(model.children())
        in_features = layers[-1].in_features
        linear = nn.Linear(in_features=in_features,
                        out_features=num_classes)
        
        model_upper_parts = nn.Sequential(*layers[:-1])
        if is_freeze:
            for param in model_upper_parts.parameters():
                param.requires_grad = False
        
        model = nn.Sequential(model_upper_parts, linear)
        return model
    

    def forward(self, x):
        x = self._preprocess(x)
        outputs = self.model(x)
        return outputs

    def _preprocess(self, img):

        # to torch.Tensor
        if isinstance(img, Image.Image):
            x = np.array(img)
            x = torch.from_numpy(x)
        elif isinstance(img, np.ndarray):
            x = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            x = img
        else:
            raise TypeError(f"image should be either PIL.Image.Image or numpy.ndarray. Input type is {type(img)}")

        # C, H, W
        if x.size()[0] != 3:
            x = x.permute(2, 0, 1)

        # dtype conversion
        if x.dtype != torch.float:
            x = x.type(torch.float)

        # dimension for batch
        if x.dim() == 3:
            x = x.unsqueeze(0)

        return x      

    def shared_step(self, batch ,stage):
        x, y = batch
        preds = self.model(x)
        if y.dim() == 2:
            y = y.squeeze(1)
        loss = self.loss_fn(preds, y.long())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True) 
        return {"loss": loss}


    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return self.optimizer


if __name__ == "__main__":
    # model = get_model("mobilenetv3_small_050", 10)

    classifier = Classfier()
    model = classifier.model

    inputs = torch.randn(5, 3, 224, 224)
    model.eval()
    start = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    print(f"[{time.time() - start :.6f}] seconds...")

    print(f"{outputs.shape}")
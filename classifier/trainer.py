import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets


from random import Random, shuffle
from data_generation import SetCardDataset
from data_generation import generate_card
from data_generation import SetCardDataset
from visual_augmentations import augment_card
from classifier import SetClassifier



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    ds = SetCardDataset(n_samples=16, base_seed=42, augment=False)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model = SetClassifier(channels=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_head = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        running_loss = 0.0

        for x, (y_shape, y_color, y_shad, y_count) in dl:
            x = x.to(device)
            y_shape = y_shape.to(device)
            y_color = y_color.to(device)
            y_shad = y_shad.to(device)
            y_count = y_count.to(device)

            out = model(x)
            loss = (
                ce_head(out["shape"], y_shape) +
                ce_head(out["color"], y_color) +
                ce_head(out["shading"], y_shad) + 
                ce_head(out["count"], y_count)
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        model.eval()
        with torch.no_grad():
            a_shape = a_color = a_shad = a_count = 0.0
            a_all = 0.0
            n = 0

            for x, (y_shape, y_color, y_shad, y_count) in dl:
                x = x.to(device)
                y_shape = y_shape.to(device)
                y_color = y_color.to(device)
                y_shad = y_shad.to(device)
                y_count = y_count.to(device)

                out = model(x)
                ps = out["shape"].argmax(1)
                pc = out["color"].argmax(1)
                pz = out["shading"].argmax(1)
                pn = out["count"].argmax(1)

                bs = (ps == y_shape)
                bc = (pc == y_color)
                bz = (pz == y_shad)
                bn = (pn == y_count)

                a_shape += bs.float().sum().item()
                a_color += bc.float().sum().item()
                a_shad += bz.float().sum().item()
                a_count += bn.float().sum().item()
                a_all += (bs & bc & bz & bn).float().sum().item()
                n += x.size(0)
        train_loss = running_loss / len(ds)
        print(
            f"epoch {epoch} | loss {train_loss:.4f} | "
            f"acc shape {a_shape/n:.3f} color {a_color/n:.3f} "
            f"shad {a_shad/n:.3f} count {a_count/n:.3f} | all {a_all/n:.3f}"
        )
                
if __name__ == "__main__":
    #print("Generating green card with 5 random augmentations")
    #card0 = generate_card(shape="squiggle", shading="solid", color="green", count=3)
    #card0.save("raw_card.png")

    main()

    
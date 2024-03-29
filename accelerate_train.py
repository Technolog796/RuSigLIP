import yaml
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import dataloader
from dataloader import SigLIPDataLoader
from models import SigmoidLoss, SigLIPModel


def run_epoch(
    model,
    criterion,
    data_loader,
    accelerator,
    epoch,
    optimizer=None,
    train_mode=True,
    log_interval=100,
):
    model.train() if train_mode else model.eval()
    total_loss = 0.0

    for batch_idx, (images, texts) in enumerate(
        tqdm(data_loader, disable=not accelerator.is_local_main_process)
    ):
        optimizer.zero_grad()

        img_emb, txt_emb = model(images, texts[0])
        loss = criterion(img_emb, txt_emb, positive=True)

        for i in range(1, len(texts)):
            img_emb, txt_emb = model(images, texts[i])
            loss += criterion(img_emb, txt_emb, positive=False)

        loss /= len(img_emb)
        total_loss += loss.item()

        if train_mode:
            accelerator.backward(loss)
            optimizer.step()

            if batch_idx % log_interval == 0 and accelerator.is_main_process:
                wandb.log({"Batch Loss": loss.item()}, commit=False)

                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log(
                            {
                                f"Gradients/{name}": wandb.Histogram(
                                    param.grad.cpu().detach().numpy()
                                )
                            },
                            commit=False,
                        )

    if accelerator.is_main_process:
        mode = "Train" if train_mode else "Test"
        wandb.log({f"{mode} Loss/EPOCH": total_loss / len(data_loader), "Epoch": epoch})
        print(f"Epoch {epoch}\t{mode} Loss: {total_loss / len(data_loader):.6f}")


def main(args):
    accelerator = Accelerator()

    train_dataset = getattr(dataloader, args["Train dataset name"])(
        **args["Train dataset parameters"]
    )
    train_loader = SigLIPDataLoader(train_dataset, **args["Dataloader parameters"])

    test_dataset = getattr(dataloader, args["Test dataset name"])(
        **args["Test dataset parameters"]
    )
    test_loader = SigLIPDataLoader(test_dataset, **args["Dataloader parameters"])

    model = SigLIPModel(**args["Model parameters"])
    criterion = SigmoidLoss(**args["Loss parameters"])
    optimizer = Adam(model.parameters(), **args["Optimizer parameters"])
    scheduler = StepLR(optimizer, **args["Scheduler parameters"])

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    for epoch in range(1, args["Train parameters"]["epochs"] + 1):
        run_epoch(
            model,
            criterion,
            train_loader,
            accelerator,
            epoch,
            optimizer,
            train_mode=True,
        )
        run_epoch(model, criterion, test_loader, accelerator, epoch, train_mode=False)
        scheduler.step()

    if args["Train parameters"]["save_model"] and accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        states = unwrapped_model.state_dict()
        torch.save(states, args["Train parameters"]["save_directory"])


if __name__ == "__main__":
    torch.manual_seed(42)

    with open("config.yml") as file:
        args = yaml.safe_load(file)

    wandb.login()
    project_name = "RuSigLIP"
    wandb.init(project=project_name, config=args, sync_tensorboard=True)

    main(args)

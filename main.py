import matplotlib.pyplot as plt
import torch

from checkpoint import CHECKPOINT_PATH, load_checkpoint, save_checkpoint
from dataset import OperatorDataset
from loss import loss
from mae import MAE
from patchify import patchify_tensor, unpatchify

DATA_DIR = "arknights-pfp-dataset/default"
IMAGE_SIZE = 256
PATCH_SIZE = 32
N = (IMAGE_SIZE // PATCH_SIZE) ** 2
LEN_KEEP = int(N * 0.25)

STEPS_PER_IMAGE = 512
LR = 1.5e-4

LOG_EVERY = 10
PLOT_EVERY = 50
LOSS_HISTORY_CAP = 10_000


def training_config():
    return {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "n": N,
        "len_keep": LEN_KEEP,
        "lr": LR,
        "steps_per_image": STEPS_PER_IMAGE,
    }


def patches_to_hwc(patches, patch_size):
    vis = patches.detach().clamp(0.0, 1.0)
    img = unpatchify(vis, patch_size)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    return img


def update_plots(global_step, image_idx, step_in_image, loss_val, loss_history, patches, pred, mask, fig, axes, loss_line, ims):
    ax_loss, ax_gt, ax_pred, ax_demo = axes

    x = list(range(len(loss_history)))
    loss_line.set_data(x, loss_history)
    ax_loss.relim()
    ax_loss.autoscale_view()

    pred_c = pred.detach().clamp(0.0, 1.0)
    mask_3d = mask.unsqueeze(-1).expand_as(patches)
    demo = patches * (1 - mask_3d) + pred_c * mask_3d

    ims[0].set_data(patches_to_hwc(patches, PATCH_SIZE))
    ims[1].set_data(patches_to_hwc(pred_c, PATCH_SIZE))
    ims[2].set_data(patches_to_hwc(demo, PATCH_SIZE))
    ax_pred.set_title(
        f"img {image_idx} {step_in_image}/{STEPS_PER_IMAGE} | "
        f"global {global_step} | loss {loss_val:.4f}"
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def train_step(model, optimizer, patches):
    optimizer.zero_grad()
    pred, mask = model(patches)
    loss_val = loss(patches, pred, mask)
    loss_val.backward()
    optimizer.step()
    return loss_val


def pick_random_image(dataset):
    return torch.randint(0, len(dataset), (1,)).item()


def run():
    config = training_config()
    dataset = OperatorDataset(DATA_DIR, image_size=IMAGE_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE(N, PATCH_SIZE, LEN_KEEP).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    global_step, loss_history, image_idx, step_in_image, images_completed = load_checkpoint(
        CHECKPOINT_PATH, model, optimizer, config, device
    )

    if image_idx is not None and step_in_image < STEPS_PER_IMAGE:
        print(f"resuming image {image_idx} at step {step_in_image}/{STEPS_PER_IMAGE}")
    elif images_completed > 0:
        print(f"resuming after {images_completed} completed images (global step {global_step})")

    model.train()
    patches = None
    image_path = None

    if image_idx is not None and step_in_image < STEPS_PER_IMAGE:
        image_path = dataset.paths[image_idx]
        batch = dataset.images[image_idx].unsqueeze(0).to(device)
        patches = patchify_tensor(batch, PATCH_SIZE)

    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    ax_loss, ax_gt, ax_pred, ax_demo = axes
    loss_line, = ax_loss.plot([], [], lw=2)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("global step")
    blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE).permute(1, 2, 0).numpy()
    ims = [
        ax_gt.imshow(blank, vmin=0.0, vmax=1.0),
        ax_pred.imshow(blank, vmin=0.0, vmax=1.0),
        ax_demo.imshow(blank, vmin=0.0, vmax=1.0),
    ]
    for ax, title in zip([ax_gt, ax_pred, ax_demo], ["GT", "Full pred", "MAE demo"]):
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    print(f"training until Ctrl+C — {STEPS_PER_IMAGE} steps per random image")

    try:
        while True:
            if image_idx is None or step_in_image >= STEPS_PER_IMAGE:
                image_idx = pick_random_image(dataset)
                step_in_image = 0
                image_path = dataset.paths[image_idx]
                batch = dataset.images[image_idx].unsqueeze(0).to(device)
                patches = patchify_tensor(batch, PATCH_SIZE)
                print(f"\nimage {images_completed + 1} block — {image_path} ({STEPS_PER_IMAGE} steps)")

            while step_in_image < STEPS_PER_IMAGE:
                loss_val = train_step(model, optimizer, patches)

                loss_history.append(loss_val.item())
                if len(loss_history) > LOSS_HISTORY_CAP:
                    loss_history = loss_history[-LOSS_HISTORY_CAP:]

                if global_step % LOG_EVERY == 0:
                    print(
                        f"  img {image_idx} {step_in_image + 1}/{STEPS_PER_IMAGE} "
                        f"(global {global_step}): loss {loss_val.item():.4f}"
                    )

                if global_step % PLOT_EVERY == 0:
                    with torch.no_grad():
                        viz_pred, viz_mask = model(patches)
                    update_plots(
                        global_step, image_idx, step_in_image + 1, loss_val.item(), loss_history,
                        patches, viz_pred, viz_mask, fig, axes, loss_line, ims,
                    )

                step_in_image += 1
                global_step += 1

            images_completed += 1
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                optimizer,
                global_step,
                loss_history,
                config,
                image_idx=None,
                step_in_image=STEPS_PER_IMAGE,
                images_completed=images_completed,
                image_path=image_path,
            )
            print(f"  finished {image_path} — saved {CHECKPOINT_PATH}")
            image_idx = None

    except KeyboardInterrupt:
        save_checkpoint(
            CHECKPOINT_PATH,
            model,
            optimizer,
            global_step,
            loss_history,
            config,
            image_idx=image_idx,
            step_in_image=step_in_image,
            images_completed=images_completed,
            image_path=image_path,
        )
        print(
            f"\ninterrupted — saved {CHECKPOINT_PATH} "
            f"(img {image_idx} step {step_in_image}/{STEPS_PER_IMAGE}, global {global_step})"
        )

    plt.ioff()
    plt.show(block=True)


if __name__ == "__main__":
    run()

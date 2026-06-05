from common import * 
from TrainSelection import TrainSelection
from Measure import Measure
from ConvNet import ConvNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name in known datasets [cifar10, imagenette]")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--global_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--out_path", type=str, default="models/default_out/", help="enter the desired output path from cwd")
    parser.add_argument("--loss_type", type=str, default="ce", help="enter the criterion for loss you would like to apply during training")
    parser.add_argument("--penalties", default=None, help="dictionary of layer names and weights")
    parser.add_argument("--train_type", type=str, default="coll", help="coll or not_coll")
    parser.add_argument("--global_bs", type=int, default=32, help="batch size to be used in training and test")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    number_used = 224 if args.dataset.lower() == "imagenette" else 32 if args.dataset.lower() == "cifar10" else 0
    
    if number_used == 0:
        raise ValueError("Selected dataset is not supported. Choose 'cifar10' or 'imagenette'.")
        
    model = ConvNet(number_used).to(device)
    
    # Setup structural penalties if training path is 'not_coll'
    if args.train_type.lower() == "not_coll" and args.penalties is None:
        p = {}
        maxPen = 5e-3
        counter = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if (counter + 1) % 3 == 0:
                    maxPen = maxPen * 2
                p[name] = maxPen
                counter += 1
    
        for lin_layer in ["lin1", "lin2", "lin3"]:
            if hasattr(model, lin_layer):
                p[lin_layer] = 1e-3
        if hasattr(model, "classifier"):
            p["classifier"] = 5e-4
            
        args.penalties = p
    
    # Initialize Trainer object
    trainer = TrainSelection(
        dataset=args.dataset,
        wd=args.wd,
        lr=args.lr, 
        global_epochs=args.global_epochs,
        model=model,
        out_path=args.out_path,
        loss_type=args.loss_type,
        penalties=args.penalties,
        train_type=args.train_type,
        global_bs=args.global_bs
    )

    MODEL_PATH = f"{args.out_path}/last.pt"

    # Post-training evaluation 
    model = ConvNet(number_used).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Neural Collapse measurements execution
    m = Measure(trainer.rotated_train_loader, number_used)
    nc1, nc3 = m.computeNC(model, num_classes=4)
    print(f"NC3 Metric Score: {nc3}")
    print(f"NC1 Metric Scores: {nc1}")
    

    m.plot_nc1_by_layer(nc1, "graphs/collapsed/nc1.pdf", MODEL_PATH)

    # test acc
    with torch.no_grad():
        total_correct = 0.0
        examples_seen = 0.0
        for data in trainer.rotated_test_loader:
            img, label = data[0].to(device), data[1].to(device)
            pred = model(img)
            pred_labels = pred.argmax(dim=1)
            total_correct += (pred_labels == label).sum().item()
            examples_seen += img.size(0)
            
        print(f"Test Accuracy: {100 * (total_correct / examples_seen):.2f}%")

    # train acc
    with torch.no_grad():
        total_correct = 0.0
        examples_seen = 0.0
        for data in trainer.rotated_train_loader:
            img, label = data[0].to(device), data[1].to(device)
            pred = model(img)
            pred_labels = pred.argmax(dim=1)
            total_correct += (pred_labels == label).sum().item()
            examples_seen += img.size(0)
            
        print(f"Train Accuracy: {100 * (total_correct / examples_seen):.2f}%")

    #  Layer Reconstructions & Visualization  
    VIS_SAVE_PATH = f"graphs/visualizations/{args.train_type}"
    
    #  mapping classes from rotation logic (0, 1, 2, 3 maps to degrees)
    classes = ["rot_0", "rot_90", "rot_180", "rot_270"]
    class_examples = defaultdict(list)

    for images, labels in trainer.rotated_train_loader:
        for img, label in zip(images, labels):
            class_examples[int(label.item())].append(img)

    def showImage(img, saveToPath, show=False, denorm=False):
        os.makedirs(os.path.dirname(saveToPath), exist_ok=True)
        if denorm:
            img = img / 2 + 0.5
        npimg = img.detach().cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.savefig(saveToPath, bbox_inches="tight", pad_inches=0)
        if show:
            plt.show()
        plt.close()

    def normalize_recon(recon):
        recon = recon.squeeze(0).cpu()
        recon = recon - recon.min()
        recon = recon / (recon.max() + 1e-8)
        return recon

    def best_channel_for_layer(act):
        channel_energy = act[0].abs().sum(dim=(1, 2))
        return int(channel_energy.argmax().item())

    def visualize_k_per_class():
        for class_id, class_name in enumerate(classes):
            if len(class_examples[class_id]) < 5:
                continue
            samples = random.sample(class_examples[class_id], k=5)
            print(f"\nProcessing Feature Maps for Class: {class_name}")

            showImage(
                vutils.make_grid(samples, nrow=5),
                f"{VIS_SAVE_PATH}/{class_name}_originals.png",
                show=True
            )

            # Dictionary structure to group layer visualization outputs cleanly and dynamically
            feature_vis_map = {i: [] for i in range(1, 10)}

            with torch.no_grad():
                for img in samples:
                    image = img.to(device).unsqueeze(0)
                    _ = model(image, save_deconv=True)

                    
                    for layer_idx in range(1, 10):
                        act_attr = f"act{layer_idx}"
                        if hasattr(model, act_attr):
                            activation_tensor = getattr(model, act_attr)
                            k = best_channel_for_layer(activation_tensor)
                            recon = model.deconv(k=k, layer=layer_idx)
                            feature_vis_map[layer_idx].append(normalize_recon(recon))

            for layer_idx, featureVis in feature_vis_map.items():
                if len(featureVis) > 0:
                    print(f"Layer {layer_idx} representations mapped.")
                    save_path = f"{VIS_SAVE_PATH}/conv{layer_idx}/{class_name}.png"
                    showImage(vutils.make_grid(featureVis, nrow=5), save_path, show=True)

    visualize_k_per_class()

if __name__ == "__main__":
    main()
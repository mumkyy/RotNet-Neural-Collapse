from common import * 
from PreProcess import PreProcess
class TrainSelection():
    def __init__(self, 
                 dataset,
                 wd,
                 lr, 
                 global_epochs,
                 model,
                 out_path, 
                 global_bs,
                 loss_type="ce",
                 penalties = None,
                 train_type: str = "coll"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataset = dataset
        self.wd = wd
        self.lr = lr
        self.global_epochs = global_epochs
        self.loss_type = loss_type
        self.model = model
        self.out_path = out_path
        self.train_type = train_type
        self.penalties = penalties
        self.global_bs = global_bs
        
        # Preprocess dataloaders
        dataloaders = PreProcess(dataset_name=self.dataset, global_bs=self.global_bs)
        self.rotated_train_loader = dataloaders.rotated_train_loader
        self.rotated_test_loader = dataloaders.rotated_test_loader
        self.train_loader = dataloaders.trainloader
        self.test_loader = dataloaders.testloader

        # Routing training execution path
        if self.train_type.lower() == "coll":
            self.trainLoop(wd=self.wd, lr=self.lr, model=self.model, outPath=self.out_path, loader=self.rotated_train_loader, lossType=self.loss_type)
        elif self.train_type.lower() == "not_coll":
            self.nc1regtrainLoop(wd=self.wd, lr=self.lr, model=self.model, outPath=self.out_path, loader=self.rotated_train_loader, penalties=self.penalties, lossType=self.loss_type)
        else:
            raise ValueError(f"Unknown training pipeline type: {self.train_type}")

    def trainLoop(self, wd, lr, model, outPath, loader, lossType):
        print("Standard Collapse Training Engine Initiated.")
        os.makedirs(outPath, exist_ok=True)
        
        criterion = nn.MSELoss() if lossType.lower() == "mse" else nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        model.train()
        for epoch in range(self.global_epochs):
            running_train_loss = 0.0
            total_correct = 0.0
            examples_seen = 0.0
            
            # Sub-batch window tracking elements
            window_loss = 0.0
            window_correct = 0.0
            window_examples = 0.0

            for i, data in enumerate(loader, 0):
                img, true_angle = data[0].to(self.device), data[1].to(self.device)
                og_label = true_angle.clone()

                if lossType.lower() == "mse":
                    true_angle = F.one_hot(true_angle, 4).float()

                optimizer.zero_grad(set_to_none=True)
                pred_angle = model(img)

                loss = criterion(input=pred_angle, target=true_angle)
                loss.backward()
                optimizer.step()

                # Optimization metric tracking
                bsz = og_label.size(0)
                running_train_loss += loss.item()
                window_loss += loss.item()
                
                pred = torch.argmax(pred_angle, dim=1)
                correct_in_batch = torch.sum(pred == og_label).item()
                
                total_correct += correct_in_batch
                window_correct += correct_in_batch
                examples_seen += bsz
                window_examples += bsz

                if (i + 1) % 20 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] Step Loss: {window_loss / 20:.3f} | Window Accuracy: {100 * (window_correct / window_examples):.2f}%')
                    window_loss = 0.0
                    window_correct = 0.0
                    window_examples = 0.0

            print(f"--- Epoch {epoch + 1} Complete. Total Train Accuracy: {100 * (total_correct / examples_seen):.2f}% ---")
            
            # Checkpoint management saving
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.global_epochs:
                filename = "last.pt" if (epoch + 1) == self.global_epochs else f"epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), os.path.join(outPath, filename))
                
        print("Training lifecycle complete.")

    def nc1regtrainLoop(self, wd, lr, model, outPath, loader, penalties, lossType):
        print("Anti-Collapse Regularization Training Engine Initiated.")
        feats = {}
        
        if penalties is not None:
            layers_with_pen = list(penalties.keys())
            for name, module in model.named_modules():
                if name in layers_with_pen:
                    # Register runtime forward hooks dynamically for layer structural profiling
                    module.register_forward_hook(
                        lambda module, input, output, name=name: feats.__setitem__(name, output)
                    )

        os.makedirs(outPath, exist_ok=True)
        criterion = nn.MSELoss() if lossType.lower() == "mse" else nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        model.train()
        for epoch in range(self.global_epochs):
            running_train_loss = 0.0

            for iteration, data in enumerate(loader, 0):
                img_batch, true_labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                model_pred = model(img_batch)

                loss_targets = F.one_hot(true_labels, 4).float() if lossType.lower() == "mse" else true_labels
                loss = criterion(input=model_pred, target=loss_targets)

                # Compute Anti-Collapse structural penalties
                if penalties is not None:
                    for layer in feats.keys():
                        if epoch == 0 and iteration == 0:
                            print(f"Hook Active & Regularizing Layer Target: {layer}")
                        
                        eps = 1e-6
                        z = feats.get(layer, None)
                        if z is None:
                            continue

                        # Apply Global Average Pooling (GAP) if targeting raw Convolutional Feature Volumes
                        if z.dim() == 4:
                            z = z.mean(dim=(2, 3))

                        B, D = z.shape 

                        # Vectorized internal class configurations mapping
                        counts = torch.bincount(true_labels, minlength=4).float()
                        sums = z.new_zeros(4, D)
                        sums.index_add_(0, true_labels, z)
                        means = sums / counts.clamp_min(1.0).unsqueeze(1) 

                        # Within-Class Scatter representation tracing (SW)
                        mu_y = means.index_select(0, true_labels)
                        trace_sw = ((z - mu_y) ** 2).sum() / float(B)

                        # Between-Class Scatter representation tracing (SB)
                        mu = z.mean(0) 
                        diff = means - mu.unsqueeze(0) 
                        trace_sb = (counts.unsqueeze(1) * ((diff) ** 2)).sum() / float(B)
                        
                        # Compute anti-collapse logs variance penalty metrics
                        nc1 = trace_sw / (trace_sb.detach() + eps)
                        penalty = -1 * (torch.log(nc1 + eps))
                        
                        w = penalties.get(layer, 0)
                        loss = loss + w * penalty

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                # Fixed: Logging intervals scaled down from 2000 to 100 for proper dataset matching
                if (iteration + 1) % 100 == 0:
                    print(f"Epoch: {epoch + 1} | Steps Sampled: {(iteration + 1):5d} | Combined Loss Proxy: {(running_train_loss / 100.0):.4f}")
                    running_train_loss = 0.0

            # Checkpoint management saving
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.global_epochs:
                filename = "last.pt" if (epoch + 1) == self.global_epochs else f"epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), os.path.join(outPath, filename))
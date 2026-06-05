from common import * 

class Measure():
    def __init__(self, loader, TARGET_RESOLUTION):
        self.loader = loader
        self.TARGET_RESOLUTION = TARGET_RESOLUTION

    #  Neural Collapse Measurements 

    # NC - 3 Metric
    def compute_nc3(self, model, M_dot):
        W = model.classifier.weight.detach()
        W_fro = torch.linalg.norm(W, ord="fro")
        M_dot_fro = torch.linalg.norm(M_dot, ord="fro")

        W_T = W.T
        W_norm = W_T / (W_fro + 1e-8)
        M_norm = M_dot / (M_dot_fro + 1e-8)

        dif = W_norm - M_norm
        nc3 = torch.linalg.norm(dif, ord="fro")
        return nc3

    # NC - 1 Helpers
    def computeMu_G(self, model, features, hook_name):
        device = next(model.parameters()).device
        print(f"computeMu_G {hook_name}", flush=True)
        
        dummy_input = torch.randn(1, 3, self.TARGET_RESOLUTION, self.TARGET_RESOLUTION).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
            
        num_channels = features[hook_name].size(1)
        layer_mean = torch.zeros(num_channels, device=device)
        total_examples = 0

        with torch.no_grad():
            for data in self.loader:
                img = data[0].to(device)
                _ = model(img)
                z = features[hook_name].detach()

                if z.dim() == 4:
                    gap_z = z.mean(dim=[2, 3])
                else:
                    gap_z = z
                
                batch_sum = torch.sum(gap_z, dim=0)
                layer_mean += batch_sum
                total_examples += img.size(0)

        layer_mean = layer_mean / total_examples
        return layer_mean

    def computeMu_C(self, model, features, hook_name, num_classes):
        device = next(model.parameters()).device
        print(f"computeMu_C {hook_name}", flush=True)
        
        dummy_input = torch.randn(1, 3, self.TARGET_RESOLUTION, self.TARGET_RESOLUTION).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
            
        num_channels = features[hook_name].size(1)
        layer_means = [torch.zeros(num_channels, device=device) for _ in range(num_classes)]
        class_counts = [0] * num_classes

        with torch.no_grad():
            for data in self.loader:
                img_batch, true_rot = data[0].to(device), data[1].to(device)
                _ = model(img_batch)

                z = features[hook_name].detach()
                if z.dim() == 4:
                    gap_z = z.mean(dim=[2, 3])
                else:
                    gap_z = z

                for idx in range(img_batch.size(0)):
                    individualFeature = gap_z[idx]
                    label = true_rot[idx].item()

                    layer_means[label] += individualFeature
                    class_counts[label] += 1

        for i in range(num_classes):
            if class_counts[i] > 0:
                layer_means[i] = layer_means[i] / class_counts[i]

        return layer_means

    def compute_sigB(self, num_classes, mu_G, mu_C, hook):
        print(f"compute_sigB {hook}", flush=True)
        device = mu_G[hook].device
        feature_dim = mu_C[0][hook].size(0)
        sigB = torch.zeros((feature_dim, feature_dim), device=device)
        
        for i in range(num_classes):
            dif = mu_C[i][hook] - mu_G[hook]
            sigB += torch.outer(dif, dif)
            
        return (1 / num_classes) * sigB

    def compute_sigW(self, mu_C, hook, model, features):
        print(f"compute_sigW {hook}", flush=True)
        device = next(model.parameters()).device
        feature_dim = mu_C[0][hook].size(0)
        sigW = torch.zeros((feature_dim, feature_dim), device=device)
        total = 0

        with torch.no_grad():
            for data in self.loader:
                imgs, true_rot = data[0].to(device), data[1].to(device)
                _ = model(imgs)

                z = features[hook].detach()
                if z.dim() == 4:
                    gap_z = z.mean(dim=[2, 3])
                else:
                    gap_z = z

                for idx in range(imgs.size(0)):
                    ind_feat = gap_z[idx]
                    label = true_rot[idx].item()
                    dif = ind_feat - mu_C[label][hook]
                    sigW += torch.outer(dif, dif)
                    total += 1

        return (1 / total) * sigW.detach()

    # Main Orchestrator
    def computeNC(self, model, num_classes):
        print("Computing NC1... ")
        device = next(model.parameters()).device
        features = {}
        hook_handles = []

        def get_activation(name, apply_relu=False):
            def hook(model, input, output):
                detached_out = output.detach()
                if apply_relu:
                    features[name] = torch.relu(detached_out)
                else:
                    features[name] = detached_out
            return hook

        bn_counter = 1
        hook_names_list = []

        # Register Hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook_name = f'bn{bn_counter}'
                handle = module.register_forward_hook(get_activation(hook_name, apply_relu=True))
                hook_handles.append(handle)
                hook_names_list.append(hook_name)
                bn_counter += 1

        for fc_layer in ['fc1', 'fc2', 'fc3']:
            if hasattr(model, fc_layer):
                handle = getattr(model, fc_layer).register_forward_hook(get_activation(fc_layer, apply_relu=True))
                hook_handles.append(handle)
                hook_names_list.append(fc_layer)

        if hasattr(model, 'classifier'):
            handle = model.classifier.register_forward_hook(get_activation('classifier', apply_relu=False))
            hook_handles.append(handle)
            hook_names_list.append('classifier')

        # Gather Means
        global_means = {}
        class_means = {c: {} for c in range(num_classes)}
        
        for hook in hook_names_list:
            global_means[hook] = self.computeMu_G(model, features, hook)
            layer_means_perClass = self.computeMu_C(model, features, hook, num_classes)
            for c in range(num_classes):
                class_means[c][hook] = layer_means_perClass[c]

        Sig_B = {}
        Sig_W = {}
        NC_1 = {}

        for hook in hook_names_list:
            Sig_B[hook] = self.compute_sigB(num_classes, global_means, class_means, hook)
            Sig_W[hook] = self.compute_sigW(class_means, hook, model, features)
            NC_1[hook] = torch.trace(Sig_W[hook]) / (torch.trace(Sig_B[hook]) + 1e-8)

        # Cleanup Hooks
        for handle in hook_handles:
            handle.remove()

        # Dynamic selection for NC3 (usually penultimate feature space before classifier)
        nc3_hook = 'fc3' if 'fc3' in hook_names_list else (hook_names_list[-2] if hook_names_list[-1] == 'classifier' and len(hook_names_list) > 1 else hook_names_list[-1])
        
        m_dot = torch.zeros((class_means[0][nc3_hook].size(0), num_classes), device=device)
        for c in range(num_classes):
            m_dot[:, c] = class_means[c][nc3_hook] - global_means[nc3_hook]

        print("Computing NC3... ")
        nc3 = self.compute_nc3(model, M_dot=m_dot)

        return NC_1, nc3

    def plot_nc1_by_layer(self, nc1, save_path, modelPath):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        x_axis_values = list(nc1.keys())[:-1]
        y_axis_values = [v.cpu().item() for v in nc1.values()][:-1]
        x_len = list(range(len(x_axis_values)))
        
        plt.figure()
        plt.semilogy(x_len, y_axis_values, "bx-")
        plt.xticks(x_len, x_axis_values, rotation=45, ha="right")
        plt.xlabel("layer")
        plt.ylabel("tr(SigW) / tr(SigB)")
        plt.title(f"NC1 - {os.path.basename(modelPath).split('.')[0]}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
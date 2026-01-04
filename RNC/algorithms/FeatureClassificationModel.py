import time
import torch
from . import Algorithm
from pdb import set_trace as breakpoint
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #
def accuracy(output, target, topk=(1,)):
    """Return list with top-k accuracies as plain floats."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # B × maxk
    pred = pred.t()                                                # maxk × B
    correct = pred.eq(target.view(1, -1).expand_as(pred))          # maxk × B

    res = []
    for k in topk:
        corr_k = correct[:k].reshape(-1).float().sum(0)
        res.append((corr_k * 100.0 / batch_size).item())           # ★
    return res


# ------------------------------------------------------------------ #
# model                                                              #
# ------------------------------------------------------------------ #
class FeatureClassificationModel(Algorithm):
    def __init__(self, opt):
        self.out_feat_keys = opt['out_feat_keys']
        super().__init__(opt)

        for name, layer in self.networks['classifier'].named_modules():
            if len(list(layer.children())) == 0:  # leaf only
                layer.register_forward_hook(debug_hook)


    # --------------------------------------------------------------
    # infrastructure
    # --------------------------------------------------------------
    def allocate_tensors(self):
        self.tensors = {
            'dataX':  torch.FloatTensor(),
            'labels': torch.LongTensor()
        }

    # --------------------------------------------------------------
    # training / eval hooks
    # --------------------------------------------------------------
    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)


    # --------------------------------------------------------------
    # core logic
    # --------------------------------------------------------------
    def process_batch(self, batch, do_train=True):
        # ---------- load batch ------------------------------------
        t0 = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX   = self.tensors['dataX']
        labels  = self.tensors['labels']
        load_tm = time.time() - t0

        # ---------- zero grad & mode ------------------------------
        finetune = self.optimizers['feat_extractor'] is not None
        if do_train:
            self.optimizers['classifier'].zero_grad()
            if finetune:
                self.optimizers['feat_extractor'].zero_grad()
            else:
                self.networks['feat_extractor'].eval()

        # ---------- forward ---------------------------------------
        feat = self.networks['feat_extractor'](
            dataX, out_feat_keys=self.out_feat_keys
        )
        if not finetune:                       # detach backbone
            with torch.no_grad():
                if isinstance(feat, (list, tuple)):
                    feat = [f.detach() for f in feat]
                else:
                    feat = feat.detach()

        pred = self.networks['classifier'](feat)

        # ---------- loss & metrics --------------------------------
        record = {}
        crit = self.criterions['loss']
        if isinstance(pred, (list, tuple)):
            loss_total = None
            for i, p in enumerate(pred):
                if isinstance(crit, torch.nn.MSELoss):
                    # one‐hot encode
                    K    = p.size(1)
                    y_oh = F.one_hot(labels, num_classes=K).float().to(p.device)
                    loss = crit(p, y_oh)
                else:
                    loss = crit(p, labels)
                loss_total = loss if loss_total is None else loss_total + loss
                record[f'prec1_c{i+1}'] = accuracy(p, labels, (1,))[0]   # ★
                record[f'prec5_c{i+1}'] = accuracy(p, labels, (5,))[0]   # ★
        else:
            p = pred
            if isinstance(crit, torch.nn.MSELoss):
                K    = p.size(1)
                y_oh = F.one_hot(labels, num_classes=K).float().to(p.device) # - 1.0 / K 
                loss_total = crit(p, y_oh)
            else:
                loss_total = crit(p, labels)

            record['prec1'] = accuracy(p, labels, (1,))[0]
            record['prec5'] = accuracy(p, labels, (5,))[0]

        record['loss'] = loss_total.item()

        if not torch.isfinite(loss_total).all():
            feat_dbg = feat[-1] if isinstance(feat, (list, tuple)) else feat
            p_dbg = p if 'p' in locals() else (pred[-1] if isinstance(pred, (list, tuple)) else pred)

            print("\n" + "="*60)
            print("NON-FINITE LOSS DETECTED")
            print("="*60)
            print(f"Feature stats - mean: {feat_dbg.mean().item():.6f}, std: {feat_dbg.std().item():.6f}, "
                f"min: {feat_dbg.min().item():.6f}, max: {feat_dbg.max().item():.6f}")
            print(f"Pred stats - min: {p_dbg.min().item():.3e}, max: {p_dbg.max().item():.3e}, mean: {p_dbg.mean().item():.3e}")
            print(f"Pred has inf: {torch.isinf(p_dbg).sum().item()} values")
            print(f"Loss value: {loss_total.item():.3e}")
            raise RuntimeError("Non-finite loss - stopping before backward")

        # ---------- backward --------------------------------------
        if do_train:
            loss_total.backward()

            self.optimizers['classifier'].step()
            if finetune:
                self.optimizers['feat_extractor'].step()

        proc_tm = time.time() - t0
        total_tm = load_tm + proc_tm
        record['load_time'] = 100 * load_tm / total_tm
        record['process_time'] = 100 * proc_tm / total_tm
        return record


def debug_hook(module, input, output):
    """Hook to print layer statistics during forward pass."""
    try:
        inp = input[0] if isinstance(input, tuple) else input
        
        # Handle None outputs
        if output is None:
            print(f"{module.__class__.__name__}: output is None")
            return
            
        # Check if finite
        inp_finite = torch.isfinite(inp).all()
        out_finite = torch.isfinite(output).all()
        
        print(f"{module.__class__.__name__:15s}: "
              f"in [{inp.min():.3e}, {inp.max():.3e}] {'✓' if inp_finite else '✗'} → "
              f"out [{output.min():.3e}, {output.max():.3e}] {'✓' if out_finite else '✗'}")
              
    except Exception as e:
        print(f"{module.__class__.__name__}: Hook error: {e}")
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import argparse 

def parse_args(): 
    p = argparse.ArgumentParser("measure and plot the stable rank of the different model layers and fro norm of layer weights")

    p.add_argument("--exp", required=True,  help="enter experiment config file path")
    p.add_argument("--exp_dir", required=True, default=None, help="enter the path to the model you want measurements on ")
    p.add_argument("--net-key", default="model")
    p.add_argument("--arch-class", default="NetworkInNetwork")
    p.add_argument(
        "--layers",
        type=str,
        default="conv2.block0,conv2.block1,conv2.block2,conv3.block0,conv3.block1,conv3.block2,conv3.block3,conv4.block0,conv4.block1,conv4.block2,conv4.block3,conv4.block4,conv4.block5,conv5.block0,conv5.block1,conv5.block2,lin1,lin2,classifier",
        help="Comma list of exposed feature keys for NC1 (e.g. conv1,conv2,conv3,conv4)",
    )
    p.add_argument("--ckpt-glob", default="model_net_epoch*", help='e.g. "model_net_epoch*"')
    return p.parse_args()

def build_fresh_model(config , net_key, arch_class):
    net_cfg_all = config.get("networks", {})
    if net_key not in net_cfg_all:
        raise RuntimeError(f"net_key '{net_key}' not in config['networks']. Keys: {list(net_cfg_all.keys())}")

    net_cfg = net_cfg_all[net_key]
    def_file = Path(net_cfg["def_file"])
    if not def_file.is_file():
        raise FileNotFoundError(f"Architecture file not found: {def_file}")

    spec_model = importlib.util.spec_from_file_location(def_file.stem, def_file)
    mod_model = importlib.util.module_from_spec(spec_model)
    assert spec_model.loader is not None
    spec_model.loader.exec_module(mod_model)

    cls_name = arch_class or net_cfg.get("arch", def_file.stem)
    if not hasattr(mod_model, cls_name):
        raise RuntimeError(f"Class '{cls_name}' not found in {def_file}")

    ModelCls = getattr(mod_model, cls_name)
    opt_dict = net_cfg.get("opt", {}).copy()

    # Your NIN expects opt dict
    model = ModelCls(opt_dict).cuda()

    
    return model



def load_state_dict(model,  ckpt_path) :
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            sd = state["state_dict"]
        elif "network" in state:
            sd = state["network"]
        elif "model" in state:
            sd = state["model"]
        else:
            sd = state
    else:
        sd = state
    model.load_state_dict(sd, strict=True)


def discover_checkpoints(exp_dir, ckpt_glob) :
    files = list(exp_dir.glob(ckpt_glob))
    if not files:
        raise RuntimeError(f"No ckpts found in {exp_dir} matching '{ckpt_glob}'")

    out = {}
    for f in files:
        stem = f.stem
        if "epoch" not in stem:
            continue
        try:
            ep = int(stem.split("epoch")[-1])
            out[ep] = f
        except ValueError:
            continue

    if not out:
        raise RuntimeError("Found checkpoint files but failed to parse epoch numbers.")
    return out



def main(): 

    args = parse_args()

    cfg_file = Path('config') / f"{args.exp}.py"
    spec = importlib.util.spec_from_file_location("cfg", cfg_file)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.config
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = Path('experiments') / args.exp
    config['exp_dir'] = str(exp_dir)

    # load the model                
    model = build_fresh_model(
        config=config,
        net_key=args.net_key,
        arch_class=args.arch_class,
    )

    # Load the checkpoint we want to measure (or the last one found)
    epoch_to_path = discover_checkpoints(exp_dir, args.ckpt_glob)
    all_epochs = sorted(epoch_to_path.keys())


    # validate layer keys exist
    layer_keys = [x.strip() for x in args.layers.split(",") if x.strip()]
    if not hasattr(model, "all_feat_names"):
        raise RuntimeError("Model missing all_feat_names (expected NetworkInNetwork).")
    for k in layer_keys:
        if k not in model.all_feat_names:
            raise RuntimeError(f"Layer key '{k}' not in model.all_feat_names: {model.all_feat_names}")

    print(model)
    print(layer_keys)
    print(all_epochs)
    load_state_dict(model, epoch_to_path[max(all_epochs)])

    # model.cuda() 
    # modules = {} 
    # model_weight_statistics = {}
    # for l in layer_keys: 
    #     model
        

    # for m in modules:
    #     w = m.weight 
    #     if isinstance(w, torch.nn.conv2d): 
    #         w.view(w.shape[0], -1)
    #     frob = w.norm(ord="fro") 

    #     model_weight_statistics[l]["froNorm"] = frob

    #     l2_sq = w.norm(ord=2) ** 2 
    #     frob_sq = frob ** 2 
    #     # defined sr(W) = ||W||F^2 / ||W||2^2 - https://arxiv.org/html/2407.21594v1 s
    #     model_weight_statistics[l]["stableRank"] =  frob_sq / l2_sq

    #     '''
    #     Stable rank is a continuous measure of matrix size defined by the ratio of the squared Frobenius norm to the squared operator (spectral) norm, whereas spectral rank is the traditional algebraic rank derived directly from the count of non-zero singular values in the matrix spectrum
    #     '''
    #     eps = 1e-4
    #     # include singular value sigma if it is larger than atol i.e. eps 
    #     rank = torch.linalg.matrix_rank(w, atol=eps, rtol=0.0)

    #     model_weight_statistics[l]["effectiveRank"] = rank


    # print(model_weight_statistics)

    return

if __name__ == "__main__":
    main()
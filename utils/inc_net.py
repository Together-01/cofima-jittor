import logging
import os
import copy
import numpy as np

import jittor as jt
import jittor.nn as nn

from network.vit import VisionTransformer


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self._head_names = []
        self._old_state_dict = None

    def update(self, nb_classes, freeze_old=True):
        if nb_classes <= 0:
            return
        if freeze_old:
            for head_name in self._head_names:
                head = getattr(self, head_name)
                for param in head.parameters():
                    try:
                        param.stop_grad()
                    except Exception:
                        pass

        head_name = "head_{}".format(len(self._head_names))
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=True)
        self._init_head(new_head)
        setattr(self, head_name, new_head)
        self._head_names.append(head_name)

    def _init_head(self, head):
        w_shape = tuple(head.weight.shape)
        trunc = np.random.normal(loc=0.0, scale=0.02, size=w_shape)
        invalid = (trunc < -0.04) | (trunc > 0.04)
        while np.any(invalid):
            trunc[invalid] = np.random.normal(loc=0.0, scale=0.02, size=int(invalid.sum()))
            invalid = (trunc < -0.04) | (trunc > 0.04)
        trunc = trunc.astype("float32")
        rand_w = jt.array(trunc)
        try:
            head.weight.assign(rand_w)
        except Exception:
            try:
                head.weight.update(rand_w)
            except Exception:
                pass

        if head.bias is not None:
            try:
                zero_b = jt.zeros(head.bias.shape)
                try:
                    head.bias.assign(zero_b)
                except Exception:
                    head.bias.update(zero_b)
            except Exception:
                pass

    def backup(self):
        self._old_state_dict = copy.deepcopy(self.state_dict())

    def recall(self):
        if self._old_state_dict is None:
            return
        try:
            self.load_state_dict(self._old_state_dict, strict=True)
        except TypeError:
            self.load_state_dict(self._old_state_dict)

    def execute(self, x):
        if len(self._head_names) == 0:
            raise RuntimeError("No classifier head initialized. Call update() first.")
        logits = []
        for head_name in self._head_names:
            head = getattr(self, head_name)
            logits.append(head(x))
        if len(logits) == 1:
            return logits[0]
        return jt.concat(logits, dim=1)


class ViTIncrementalModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        name = model_name.lower()
        model_alias = {
            "cofima_cifar_mocov3": "vit-b-p16-mocov3",
        }
        name = model_alias.get(name, name)
        if name not in {"vit-b-p16", "vit-b-p16-mocov3"}:
            raise ValueError(
                "Only ViT models are supported: vit-b-p16, vit-b-p16-mocov3, or cofima_cifar_mocov3"
            )
        self.convnet = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )
        self.fc = SimpleContinualLinear(self.convnet.feature_dim)

    @property
    def feature_dim(self):
        return self.convnet.feature_dim

    def update_fc(self, nb_classes):
        self.fc.update(nb_classes, freeze_old=True)

    def execute(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            logits = self.fc(x)
            return {"logits": logits}

        if bcb_no_grad:
            with jt.no_grad():
                feats = self.convnet(x)
        else:
            feats = self.convnet(x)

        logits = self.fc(feats)
        return {"features": feats, "logits": logits}


def load_pretrained_backbone(model, path):
    if not path:
        return
    if not os.path.exists(path):
        logging.warning("Pretrained not found: {}".format(path))
        return
    state = jt.load(path)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "module" in state and isinstance(state["module"], dict):
            state = state["module"]

    if not isinstance(state, dict):
        logging.warning("Unsupported pretrained format: {}".format(type(state)))
        return

    model_state = model.state_dict()
    prefixes = ["module.base_encoder.", "base_encoder.", "module."]
    adapted = {}
    loaded_extra = 0

    def _to_jt_var(value):
        if isinstance(value, jt.Var):
            return value
        try:
            return jt.array(value)
        except Exception:
            try:
                return jt.array(value.numpy())
            except Exception:
                return value

    def _assign_var(var, value):
        v = _to_jt_var(value)
        try:
            var.assign(v)
        except Exception:
            try:
                var.update(v)
            except Exception:
                pass
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        if new_key.startswith("head") or new_key.startswith("fc"):
            continue
        if new_key in {"cls_token", "pos_embed"} and hasattr(model, new_key):
            _assign_var(getattr(model, new_key), value)
            loaded_extra += 1
            continue
        if new_key in model_state and tuple(model_state[new_key].shape) == tuple(value.shape):
            adapted[new_key] = value

    try:
        if len(adapted) == 0:
            logging.warning("No matched pretrained keys found in {}".format(path))
            return
        try:
            model.load_state_dict(adapted, strict=False)
        except TypeError:
            model.load_state_dict(adapted)
        logging.info(
            "Loaded pretrained from {} (matched {}/{})".format(
                path, len(adapted) + loaded_extra, len(model_state)
            )
        )
        if len(adapted) + loaded_extra < max(1, len(model_state) // 2):
            logging.warning(
                "Pretrained match ratio is low: {}/{}".format(
                    len(adapted) + loaded_extra, len(model_state)
                )
            )
    except Exception as exc:
        logging.warning("Failed to load pretrained: {}".format(exc))

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=16, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        if r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_A = nn.Parameter(torch.zeros((r, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += lora_out
        return result


def inject_lora_to_vit(model, r=16):
    replace_cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential) and ('mlp' in name or 'c_fc' in name or 'c_proj' in name):
            for idx, child in module.named_children():
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child, r=r)
                    setattr(module, str(idx), lora_layer)
                    replace_cnt += 1
    print("==> Successfully injected {} LoRA layers.".format(replace_cnt))
    return model


def mark_only_lora_as_trainable(model):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        elif 'prompt_learner' in name:
            param.requires_grad = True
        elif 'classifier' in name or 'bottleneck' in name or 'bnneck' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_vit_top_layers(model, num_layers=4):
    """Unfreeze the top N layers of ViT transformer for full fine-tuning.
    This allows the model to actually adapt its high-level visual features,
    going far beyond what LoRA's low-rank perturbation can achieve."""
    unfrozen = 0
    total_layers = 12
    start_layer = total_layers - num_layers

    for name, param in model.named_parameters():
        for layer_idx in range(start_layer, total_layers):
            layer_key = 'transformer.resblocks.{}.'.format(layer_idx)
            if layer_key in name:
                param.requires_grad = True
                unfrozen += 1
                break

    print("==> Unfroze ViT layers {}-{} ({} params)".format(
        start_layer, total_layers - 1, unfrozen))
    return model

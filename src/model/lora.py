from torch import nn
import torch

class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
        super(LoRALinear, self).__init__()

        self.base_linear = base_linear

        for p in base_linear.parameters():
            p.requires_grad = False
        
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features #d_model, d_model이 아닌 경우에도 사용 가능하도록 일반화

        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        nn.init.zeros_(self.lora_b.weight)

        self.scaling = alpha / rank

    def forward(self, input_feature):
        base_output = self.base_linear(input_feature)

        lora_feature = self.lora_a(input_feature)
        lora_output = self.lora_b(lora_feature)

        scaled_output = lora_output * self.scaling

        return base_output + scaled_output

def load_baseline_weights_into_lora_model(lora_model, baseline_path, device):
    checkpoint = torch.load(baseline_path, map_location=device)
    baseline_state = checkpoint["model_state_dict"]

    lora_state = lora_model.state_dict()
    new_state = {}

    loaded_keys = []
    skipped_keys = []

    for lora_key in lora_state.keys():
        # LoRA adapter 자체는 baseline에 없으므로 그대로 초기값 유지
        if "lora_a" in lora_key or "lora_b" in lora_key:
            skipped_keys.append(lora_key)
            continue

        # LoRA의 base_linear는 baseline의 원래 linear weight/bias에 대응
        baseline_key = lora_key.replace(".base_linear", "")

        if baseline_key in baseline_state:
            if lora_state[lora_key].shape == baseline_state[baseline_key].shape:
                new_state[lora_key] = baseline_state[baseline_key]
                loaded_keys.append((lora_key, baseline_key))
            else:
                skipped_keys.append(lora_key)
        elif lora_key in baseline_state:
            if lora_state[lora_key].shape == baseline_state[lora_key].shape:
                new_state[lora_key] = baseline_state[lora_key]
                loaded_keys.append((lora_key, lora_key))
            else:
                skipped_keys.append(lora_key)
        else:
            skipped_keys.append(lora_key)

    lora_state.update(new_state)
    lora_model.load_state_dict(lora_state, strict=False)

    print(f"Baseline weights loaded into LoRA model from: {baseline_path}")
    print(f"Loaded param count: {len(loaded_keys)}")
    print(f"Skipped param count: {len(skipped_keys)}")

    return checkpoint.get("epoch", None)
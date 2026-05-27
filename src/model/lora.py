from torch import nn

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


import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=1, padding_idx=0):
        super().__init__()
     
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.hidden_dim = hidden_dim

        
        self.W_i = nn.Linear(embed_dim, hidden_dim)
        # 初始化输入门参数，U_i：上一个隐状态至隐状态
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        # 初始化遗忘门参数，W_f：输入至隐状态
        self.W_f = nn.Linear(embed_dim, hidden_dim)
        # 初始化遗忘门参数，U_f：上一个隐状态至隐状态
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        # 初始化输出门参数，W_o：输入至隐状态
        self.W_o = nn.Linear(embed_dim, hidden_dim)
        #初始化输出门参数，U_o：上一个隐状态至隐状态
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

        # 初始化候选状态参数，W_c：输入至隐状态
        self.W_c = nn.Linear(embed_dim, hidden_dim)
        #初始化候选状态参数，U_c：上一个隐状态至隐状态
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        # 全连接层将最终的隐藏状态映射到 embed_dim（目标嵌入空间）
        self.fc = nn.Linear(hidden_dim, embed_dim)
    
    
    #请完成forward函数
    def forward(self, captions: torch.Tensor):
        """
        captions: [B, T]，表示批次 B 中每个句子的 token id 序列，T 为序列长度。
        """

        B, T = captions.size()
        x = self.embedding(captions)  # [B, T, embed_dim]

        # 初始化输出隐状态和细胞状态
        h = torch.zeros(B, self.hidden_dim, device=captions.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_dim, device=captions.device, dtype=x.dtype)

        # 生成mask，忽略padding_idx的token
        mask = (captions != self.embedding.padding_idx)  # [B, T]

        for t in range(T):
            x_t = x[:, t, :]  # 当前时间步的输入 [B, embed_dim]

            g_t = torch.tanh(self.W_c(x_t) + self.U_c(h))     # 输入节点
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h))  # 输入门
            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h))  # 遗忘门
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h))  # 输出门

            # 仅更新非padding位置
            mask_t = mask[:, t].unsqueeze(1).type_as(h)  # [B, 1]
            c = i_t * g_t + f_t * c
            c = mask_t * c + (1 - mask_t) * c.detach()
            h = o_t * torch.tanh(c)
            h = mask_t * h + (1 - mask_t) * h.detach()

        # 将最终隐状态 h 通过全连接层映射至目标嵌入空间
        out = self.fc(h) # [B, embed_dim]
        # 对输出结果进行 L2 正则化归一化，确保嵌入向量单位化
        return F.normalize(out, p=2, dim=1)

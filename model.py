import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from tqdm import tqdm
torch.manual_seed(1337)
#------Hyperparameters----------#
device = "cuda" if torch.cuda.is_available() else "cpu"
learnr = 5e-4
num_digits = 10
max_len = (num_digits*2) + (num_digits+1)
batch_size = 256
max_iters = 1000
eval_iters = 10
n_embd = 128
n_heads = 8
dropout = 0.1

num_samples = 400000
#--------tokenizer----#
vocab = {str(i): i for i in range(10)}
itos = {i:s for i,s in enumerate(vocab)}
encode = lambda e : [vocab[eq] for eq in e]
decode = lambda d : ''.join(itos[eq] for eq in d)
vocab_size = len(vocab)
#-----------#--------#
def make_dataset():
    dataset = []
    for i in range(num_samples):
        n1 = random.randint(0,(10**num_digits-1))
        n2 = random.randint(0,(10**num_digits-1))
        res = n1 + n2
        n1 = f'%0{num_digits}d' % n1
        n2 = f'%0{num_digits}d' % n2
        res = f'%0{num_digits+1}d' % res
        res = str(res)
        input_str = f'{n1}{n2}{res[::-1]}'
        dataset.append(input_str)
    return dataset

dataset = make_dataset()
n_split = int(0.8 * len(dataset))
train_data = dataset[:n_split]
val_data = dataset[n_split:]

def get_batch(split):
    input_batch = []
    target_batch = []
    for _ in range(batch_size):
        if split=="train":
            data = train_data
        else:
            data = val_data
        input_str = random.choice(data)
        
        input_tokens = encode(input_str[:-1])
        target_tokens = encode(input_str[1:])
        target_tokens[:(num_digits*2-1)] = [-1] * ((num_digits*2)-1)
        input_batch.append(input_tokens)
        target_batch.append(target_tokens)
    input_batch = torch.tensor(input_batch,dtype = torch.long,device = device)
    target_batch = torch.tensor(target_batch,dtype=torch.long,device = device)
    return input_batch,target_batch

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer('tril',torch.tril(torch.ones(max_len-1,max_len-1)))

    def forward(self,x):
        
        B,T,C = x.shape #B = Batch , T = Time or block_size , C = channels or n_embd
        key = self.key(x)      #B,block_size , n_embd @ n_embd,head_size -->B,block_size,head_size
        query = self.query(x)  #B,block_size , n_embd @ n_embd,head_size -->B,block_size,head_size
        value = self.value(x)  #B,block_size , n_embd @ n_embd,head_size -->B,block_size,head_size
        wei = query @ key.transpose(-2,-1) * C**-0.5   # B,block_size,head_size @ B,head_size,block_size --> B,block_size,block_size --> quadratic dependency on context length
        wei = wei.masked_fill(self.tril[:T,:T]==0 ,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ value
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
    def forward(self,x):
        out = self.proj(torch.cat([h(x) for h in (self.heads)],dim=-1))
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.ffn = nn.Sequential(
                nn.Linear(n_embd,4 *n_embd),
                nn.ReLU(),
                nn.Linear(4*n_embd,n_embd),
                nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.ffn(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.mattn = MultiHeadAttention(n_heads,head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.mattn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class addGPT(nn.Module):
    def __init__(self,train=True):
        super().__init__()
        self.train_mode = train  # Flag to control behavior
        self.tok_embeddings_table = nn.Embedding(vocab_size,n_embd)
        self.pos_embeddings_table = nn.Embedding(max_len - 1,n_embd)
        # self.attnHead = Head(n_embd)
        self.blocks = nn.Sequential(
                Block(n_embd,n_heads),
                Block(n_embd,n_heads),
                nn.LayerNorm(n_embd)
        )
        # self.mattn = MultiHeadAttention(4,n_embd//4)
        # self.feedforward = FeedForward(n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_embeddings_table(idx)  # B,T,n_embd
        pos_emb = self.pos_embeddings_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # B,T,vocab_size where C = vocab_size
         
        if targets is None or not self.train_mode:
            loss = None
        else:
            B,T,C = logits.shape
            
            logits = logits.contiguous().view(B*T,C)
            targets = targets.contiguous().view(B*T)
            loss = F.cross_entropy(logits,targets,ignore_index = -1)  # Select only masked targets 
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -max_len:]
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
@torch.no_grad()
def get_loss():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for split in ["train", "val"]:
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learnr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[900], gamma=0.5)
    
    progress_bar = tqdm(range(max_iters), 
                       desc="Training Progress",
                       unit="iter")
    
    for i in progress_bar:
        x, y = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % eval_iters == 0 or i == max_iters - 1:
            out = get_loss()
            train_loss = out["train"]
            val_loss = out["val"]
            progress_bar.set_description(f"Training Progress (Loss: {val_loss:.4f})")
            # Update progress bar with more detailed metrics
            progress_bar.set_postfix({
                'Step': f"{i}/{max_iters}",
                'Train Loss': f"{train_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}"
            })
    
    
    torch.save(model.state_dict(), 'addition_weights.pth')
    print("\nTraining completed")

if __name__ == "__main__":
    model = addGPT(train=True)  # Training mode
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print(f'Model size: {sum(p.numel() for p in model.parameters())}')
    train()


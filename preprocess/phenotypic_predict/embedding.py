import os
import pandas as pd
import torch
from transformers.models.bert.configuration_bert import BertConfig
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Config / model names
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
FASTA_PATH = "./data/phenotypic_predict_seq.fasta"
OUT_CSV = "./data/all_data_embeddings_tensor.csv"
BATCH_SIZE = 8  # adjust to your GPU memory
MAX_LENGTH = 512  # set according to model; change if needed

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model + tokenizer
config = BertConfig.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, config=config, add_pooling_layer=False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()  # set eval mode

# collect sequences and ids
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
if len(records) == 0:
    raise ValueError(f"No sequences found in {FASTA_PATH}")

seqs = [str(r.seq) for r in records]
ids = [r.id for r in records]

embeddings_list = []

# helper for masked mean pooling
def masked_mean(hidden_states, attention_mask):
    # hidden_states: (batch, seq_len, hidden_dim)
    # attention_mask: (batch, seq_len)
    attn = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
    hidden_states = hidden_states * attn  # zero out padded tokens
    sum_embeddings = hidden_states.sum(dim=1)  # (batch, hidden_dim)
    lengths = attn.sum(dim=1).clamp(min=1e-9)  # avoid div by zero
    return sum_embeddings / lengths

# process in batches
for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Encoding"):
    batch_seqs = seqs[i : i + BATCH_SIZE]
    # tokenize with truncation and padding to max length in batch
    encoded = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs[0] is last_hidden_state: (batch, seq_len, hidden_dim)
        last_hidden = outputs[0]

        # masked mean pooling using attention_mask
        pooled = masked_mean(last_hidden, attention_mask)  # (batch, hidden_dim)
        embeddings_list.append(pooled.cpu())  # move to CPU to save GPU memory

# concatenate all batches
embeddings_tensor = torch.cat(embeddings_list, dim=0)  # (N, hidden_dim)

# Convert to pandas DataFrame
emb_np = embeddings_tensor.detach().numpy()  # already on CPU
df = pd.DataFrame(emb_np, index=ids)

# Save
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=True)
print(f"Saved embeddings to {OUT_CSV}")

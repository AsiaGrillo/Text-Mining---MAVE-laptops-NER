"""
MAVE Laptops NER Dashboard
==========================
Run with: python dashboard.py
Then open: http://127.0.0.1:8050
"""

import os
import re
import json
import torch
import numpy as np
from collections import Counter, defaultdict
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash import ctx as dash_ctx
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = '/Users/asiagrillo/Desktop/MAVE/'
GLOVE_PATH = '/Users/asiagrillo/Desktop/MAVE/glove.6B.300d.txt'

PATHS = {
    'train_orig':  DATA_DIR + 'laptops_train.jsonl',
    'train_clean': DATA_DIR + 'laptops_train_rag_cleaned.jsonl',
    'test_orig':   DATA_DIR + 'laptops_test.jsonl',
    'test_clean':  DATA_DIR + 'laptops_test_rag_cleaned.jsonl',
}

WEIGHTS = {
    'bilstm_orig':   DATA_DIR + 'bilstm_enhanced_original.pt',
    'bilstm_clean':  DATA_DIR + 'bilstm_enhanced_cleaned.pt',
    'deberta_orig':  DATA_DIR + 'deberta_v2_original.pt',
    'deberta_clean': DATA_DIR + 'deberta_v2_cleaned.pt',
}

# ── Entity classes ─────────────────────────────────────────────────────────────
ENTITY_CLASSES = ['BRAND', 'SCREEN_SIZE', 'PROCESSOR', 'RESOLUTION', 'BATTERY']
LABEL2ID = {'O': 0}
for ent in ENTITY_CLASSES:
    LABEL2ID[f'B-{ent}'] = len(LABEL2ID)
    LABEL2ID[f'I-{ent}'] = len(LABEL2ID)
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ── Okabe-Ito palette ──────────────────────────────────────────────────────────
ENTITY_COLORS = {
    'BRAND':       '#56B4E9',
    'SCREEN_SIZE': '#E69F00',
    'PROCESSOR':   '#009E73',
    'RESOLUTION':  '#CC79A7',
    'BATTERY':     '#0072B2',
}
COLORS = {
    'blue':   '#56B4E9',
    'orange': '#E69F00',
    'green':  '#009E73',
    'pink':   '#CC79A7',
    'dblue':  '#0072B2',
    'red':    '#D55E00',
}

# ── Dark theme ─────────────────────────────────────────────────────────────────
P = {
    'bg':      '#0F1117',
    'card':    '#1A1D27',
    'sidebar': '#141720',
    'border':  '#2A2D3A',
    'text':    '#E2E8F0',
    'muted':   '#64748B',
    'accent':  '#56B4E9',
    'active':  '#1E3A5F',
}

# ── Results ────────────────────────────────────────────────────────────────────
RESULTS = {
    'BiLSTM\nOrig→Orig':    {'micro_f1': 0.7346, 'BRAND': 0.6589, 'SCREEN_SIZE': 0.8252, 'PROCESSOR': 0.6979, 'RESOLUTION': 0.5561, 'BATTERY': 0.8889},
    'BiLSTM\nClean→Orig':   {'micro_f1': 0.5293, 'BRAND': 0.0026, 'SCREEN_SIZE': 0.8185, 'PROCESSOR': 0.4453, 'RESOLUTION': 0.5622, 'BATTERY': 0.7368},
    'BiLSTM\nClean→Clean':  {'micro_f1': 0.7518, 'BRAND': 0.0073, 'SCREEN_SIZE': 0.8419, 'PROCESSOR': 0.7847, 'RESOLUTION': 0.6974, 'BATTERY': 0.7778},
    'DeBERTa\nOrig→Orig':   {'micro_f1': 0.7456, 'BRAND': 0.6718, 'SCREEN_SIZE': 0.8282, 'PROCESSOR': 0.7234, 'RESOLUTION': 0.5963, 'BATTERY': 0.8889},
    'DeBERTa\nClean→Orig':  {'micro_f1': 0.5346, 'BRAND': 0.0357, 'SCREEN_SIZE': 0.8246, 'PROCESSOR': 0.4502, 'RESOLUTION': 0.6075, 'BATTERY': 0.8421},
    'DeBERTa\nClean→Clean': {'micro_f1': 0.7689, 'BRAND': 0.1937, 'SCREEN_SIZE': 0.8479, 'PROCESSOR': 0.8095, 'RESOLUTION': 0.7293, 'BATTERY': 0.8889},
}

BILSTM_PARAMS = [
    ('Embeddings',     'GloVe 300d + Char-CNN'),
    ('Hidden dim',     '256 (bidirectional)'),
    ('Batch size',     '32'),
    ('Learning rate',  '5e-4'),
    ('Optimizer',      'Adam (wd=1e-4)'),
    ('Epochs',         '30'),
    ('Early stopping', 'patience=5'),
    ('LR scheduler',   'StepLR (step=10, γ=0.5)'),
    ('Grad clip',      '5.0'),
    ('Output layer',   'CRF'),
]

DEBERTA_PARAMS = [
    ('Base model',     'microsoft/deberta-v3-base'),
    ('Tokenizer',      'SentencePiece (WordPiece)'),
    ('Max length',     '128 subword tokens'),
    ('Batch size',     '16'),
    ('Learning rate',  '2e-5'),
    ('Optimizer',      'AdamW (wd=0.01)'),
    ('Epochs',         '15'),
    ('Early stopping', 'patience=5'),
    ('Warmup',         '10% of total steps'),
    ('Grad clip',      '1.0'),
    ('Output layer',   'Linear classifier'),
]

HISTORY = {
    'bilstm_orig':  {'train_loss': [10.66,7.24,6.61,6.15,5.66,5.20,4.75,4.33], 'val_f1': [0.6877,0.7404,0.7470,0.7407,0.7433,0.7444,0.7397,0.7452]},
    'bilstm_clean': {'train_loss': [8.59,5.33,4.77,4.37,3.95,3.56,3.21,2.93],  'val_f1': [0.6928,0.7345,0.7477,0.7448,0.7420,0.7439,0.7436,0.7396]},
    'deberta_orig':  {'train_loss': [0.859,0.486,0.431,0.406,0.388,0.372,0.357,0.345,0.332,0.319,0.309,0.299], 'val_f1': [0.6525,0.7256,0.7378,0.7389,0.7395,0.7434,0.7476,0.7416,0.7411,0.7398,0.7443,0.7407]},
    'deberta_clean': {'train_loss': [0.728,0.363,0.319,0.297,0.284,0.268,0.254,0.243,0.234,0.224,0.213,0.206], 'val_f1': [0.6607,0.7174,0.7411,0.7412,0.7400,0.7410,0.7492,0.7410,0.7451,0.7406,0.7448,0.7391]},
}

CM_DATA = {
    'bilstm_orig':   np.array([[425,7,3,0,0,328],[1,727,0,0,0,204],[6,4,298,1,0,165],[2,2,0,52,0,62],[0,0,0,0,8,2]]),
    'bilstm_clean':  np.array([[3,1,26,0,0,244],[1,712,0,0,0,193],[0,1,663,0,0,218],[0,0,1,54,0,30],[0,0,0,0,8,1]]),
    'deberta_orig':  np.array([[439,7,1,1,0,315],[4,728,0,0,0,200],[7,3,302,1,0,161],[1,1,0,65,0,51],[0,0,0,0,8,2]]),
    'deberta_clean': np.array([[34,2,28,1,0,209],[1,733,0,0,0,171],[3,0,701,0,0,178],[0,0,0,66,0,19],[0,0,0,0,8,1]]),
}
CM_COLS = ENTITY_CLASSES + ['O']

SEED = 42
DEVICE = torch.device('cpu')
MAX_WORD_LEN = 30
EMBED_DIM    = 300
HIDDEN_DIM   = 256

EXAMPLE_TEXTS = [
    'Dell XPS 15 15.6 inch FHD 1920x1080 Intel Core i7-1165G7',
    'Apple MacBook Pro 13 M1 Retina 2560x1600 18hr battery',
    'ASUS ROG Zephyrus 17.3 QHD 2560x1440 AMD Ryzen 9 5900HX',
    'HP Pavilion 15 FHD 1920x1080 Intel Core i3 10th Gen',
    'Lenovo ThinkPad X1 Carbon 14 2560x1440 Intel i5 11th Gen',
]
# ── Data utilities ─────────────────────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def normalize_resolution(token):
    m = re.match(r'^(\d{3,4})\s*[xX×]\s*(\d{3,4})$', token)
    return f'{m.group(1)}x{m.group(2)}' if m else token

def extract_entities(tokens, labels):
    entities = []
    current_type, current_tokens, current_start = None, [], 0
    for i, (tok, lab) in enumerate(zip(tokens, labels)):
        if lab.startswith('B-'):
            if current_type:
                entities.append((current_type, ' '.join(current_tokens), current_start, i-1))
            current_type = lab[2:]; current_tokens = [tok]; current_start = i
        elif lab.startswith('I-') and current_type:
            current_tokens.append(tok)
        else:
            if current_type:
                entities.append((current_type, ' '.join(current_tokens), current_start, i-1))
            current_type, current_tokens = None, []
    if current_type:
        entities.append((current_type, ' '.join(current_tokens), current_start, len(tokens)-1))
    return entities

def span_counts(data):
    counts = {cls: 0 for cls in ENTITY_CLASSES}
    for r in data:
        for lab in r['labels']:
            if lab.startswith('B-'):
                cls = lab[2:]
                if cls in counts:
                    counts[cls] += 1
    return counts

def top_tokens_for_class(data, cls, n=10):
    counter = Counter()
    for r in data:
        for etype, val, _, _ in extract_entities(r['tokens'], r['labels']):
            if etype == cls:
                counter[val.lower()] += 1
    return counter.most_common(n)

# ── Load data ──────────────────────────────────────────────────────────────────
print('Loading data...', end=' ', flush=True)
try:
    train_orig  = load_jsonl(PATHS['train_orig'])
    train_clean = load_jsonl(PATHS['train_clean'])
    DATA_LOADED = True
    print('done.')
except Exception as e:
    train_orig = train_clean = []
    DATA_LOADED = False
    print(f'ERROR: {e}')

# ── CRF ────────────────────────────────────────────────────────────────────────
class CRF(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.transitions       = torch.nn.Parameter(torch.randn(num_labels, num_labels) * 0.1)
        self.start_transitions = torch.nn.Parameter(torch.randn(num_labels) * 0.1)
        self.end_transitions   = torch.nn.Parameter(torch.randn(num_labels) * 0.1)

    def decode(self, emissions, mask):
        viterbi = self.start_transitions + emissions[:, 0]
        bps = []
        for t in range(1, emissions.size(1)):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=1)
            viterbi = best_scores + emissions[:, t]
            bps.append(best_tags)
        viterbi += self.end_transitions
        _, best_last = viterbi.max(dim=1)
        paths = []
        for b in range(emissions.size(0)):
            path = [best_last[b].item()]
            for bp in reversed(bps):
                path.append(bp[b, path[-1]].item())
            paths.append(list(reversed(path)))
        return paths

class CharCNN(torch.nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim=30, num_filters=50, kernel_sizes=(2,3), dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs     = torch.nn.ModuleList([torch.nn.Conv1d(char_embed_dim, num_filters, ks) for ks in kernel_sizes])
        self.dropout   = torch.nn.Dropout(dropout)
        self.out_dim   = num_filters * len(kernel_sizes)
    def forward(self, char_ids):
        bsz, seq_len, mwl = char_ids.shape
        x = self.dropout(self.embedding(char_ids.view(bsz*seq_len, mwl))).permute(0,2,1)
        return torch.cat([torch.relu(conv(x)).max(dim=-1).values for conv in self.convs], dim=-1).view(bsz, seq_len, -1)

class BiLSTMEnhanced(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, char_vocab_size, pretrained_embeddings=None, dropout=0.3):
        super().__init__()
        self.word_emb = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_emb.weight.data.copy_(torch.tensor(pretrained_embeddings, dtype=torch.float32))
        self.char_cnn = CharCNN(char_vocab_size, dropout=dropout)
        self.dropout  = torch.nn.Dropout(dropout)
        self.bilstm   = torch.nn.LSTM(embed_dim + self.char_cnn.out_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.fc  = torch.nn.Linear(2*hidden_dim, num_labels)
        self.crf = CRF(num_labels)
    def decode(self, word_ids, char_ids, mask):
        x = torch.cat([self.dropout(self.word_emb(word_ids)), self.char_cnn(char_ids)], dim=-1)
        x, _ = self.bilstm(x)
        return self.crf.decode(self.fc(self.dropout(x)), mask)

# ── Model loading ──────────────────────────────────────────────────────────────
_models = {}; _vocab = {}; _glove_mat = None

def _build_vocab():
    global _vocab
    if _vocab: return
    PAD_TOKEN='<PAD>'; UNK_TOKEN='<UNK>'; PAD_CHAR='<PAD_C>'; UNK_CHAR='<UNK_C>'
    word_counter = Counter()
    for r in train_orig: word_counter.update(t.lower() for t in r['tokens'])
    word2id = {PAD_TOKEN:0, UNK_TOKEN:1}
    for w,_ in word_counter.most_common(): word2id[w] = len(word2id)
    char_set = set()
    for r in train_orig:
        for t in r['tokens']: char_set.update(t)
    char2id = {PAD_CHAR:0, UNK_CHAR:1}
    for c in sorted(char_set): char2id[c] = len(char2id)
    _vocab = {'word2id':word2id,'char2id':char2id,'PAD_TOKEN':PAD_TOKEN,'UNK_TOKEN':UNK_TOKEN,'PAD_CHAR':PAD_CHAR,'UNK_CHAR':UNK_CHAR}

def _load_glove():
    global _glove_mat
    if _glove_mat is not None: return
    _build_vocab()
    matrix = np.zeros((len(_vocab['word2id']), EMBED_DIM), dtype=np.float32)
    with open(GLOVE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word  = parts[0].lower()
            if word in _vocab['word2id']:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == EMBED_DIM:
                    matrix[_vocab['word2id'][word]] = vec
    _glove_mat = matrix

def get_model(name):
    if name in _models: return _models[name]
    if name.startswith('bilstm'):
        _build_vocab(); _load_glove()
        model = BiLSTMEnhanced(
            vocab_size=len(_vocab['word2id']), embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM, num_labels=NUM_LABELS,
            char_vocab_size=len(_vocab['char2id']),
            pretrained_embeddings=_glove_mat, dropout=0.3,
        ).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS[name], map_location=DEVICE))
        model.eval()
    else:
        import warnings; warnings.filterwarnings('ignore')
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import logging as hf_logging; hf_logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
        m   = AutoModelForTokenClassification.from_pretrained(
            'microsoft/deberta-v3-base', num_labels=NUM_LABELS,
            id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
        ).to(DEVICE).float()
        m.load_state_dict(torch.load(WEIGHTS[name], map_location=DEVICE))
        m.eval()
        model = (m, tok)
    _models[name] = model
    return model

def predict_bilstm(model_name, tokens):
    model = get_model(model_name)
    word2id = _vocab['word2id']; char2id = _vocab['char2id']
    toks = [normalize_resolution(t) for t in tokens]

    wids = torch.tensor([[word2id.get(t.lower(), word2id[_vocab['UNK_TOKEN']]) for t in toks]], dtype=torch.long).to(DEVICE)
    cids = torch.tensor([[[char2id.get(c, char2id[_vocab['UNK_CHAR']]) for c in t[:MAX_WORD_LEN]] + [char2id[_vocab['PAD_CHAR']]]*max(0,MAX_WORD_LEN-len(t)) for t in toks]], dtype=torch.long).to(DEVICE)
    mask = torch.ones(1, len(toks), dtype=torch.bool).to(DEVICE)
    with torch.no_grad():
        preds = model.decode(wids, cids, mask)[0]
    return [ID2LABEL.get(p,'O') for p in preds[:len(tokens)]]

def predict_deberta(model_name, tokens):
    model, tokenizer = get_model(model_name)
    toks = [normalize_resolution(t) for t in tokens]
    enc  = tokenizer(toks, is_split_into_words=True, max_length=128, truncation=True, return_tensors='pt')
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        preds = model(input_ids=enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE)).logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    word_preds = {}
    for i, wid in enumerate(word_ids):
        if wid is not None and wid not in word_preds:
            word_preds[wid] = ID2LABEL.get(preds[i], 'O')
    return [word_preds.get(i,'O') for i in range(len(tokens))]

def predict(model_name, tokens):
    if model_name.startswith('bilstm'): return predict_bilstm(model_name, tokens)
    return predict_deberta(model_name, tokens)
# ── Figure generators ──────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor=P['card'], plot_bgcolor=P['card'],
    font=dict(color=P['text'], family='monospace'),
)

def fig_label_dist():
    if not DATA_LOADED: return go.Figure()
    counts = span_counts(train_orig)
    fig = go.Figure(go.Bar(
        x=list(counts.keys()), y=list(counts.values()),
        marker_color=list(ENTITY_COLORS.values()),
        text=[f'{v:,}' for v in counts.values()],
        textposition='outside', textfont=dict(color=P['text']),
    ))
    fig.update_layout(**PLOTLY_DARK, height=280, showlegend=False,
                      yaxis=dict(gridcolor=P['border'], title='Span count'),
                      xaxis=dict(gridcolor=P['border']),
                      margin=dict(t=40, b=20, l=40, r=20))
    return fig

def fig_cooccurrence():
    if not DATA_LOADED: return go.Figure()
    cooc = np.zeros((5,5), dtype=int)
    for r in train_orig:
        present = set()
        for l in r['labels']:
            if l.startswith('B-') and l[2:] in ENTITY_CLASSES:
                present.add(l[2:])
        for e1 in present:
            for e2 in present:
                cooc[ENTITY_CLASSES.index(e1)][ENTITY_CLASSES.index(e2)] += 1
    fig = go.Figure(go.Heatmap(
        z=cooc, x=ENTITY_CLASSES, y=ENTITY_CLASSES,
        colorscale=[[0, P['card']], [1, COLORS['dblue']]],
        text=cooc, texttemplate='%{text}', textfont=dict(size=11),
    ))
    fig.update_layout(**PLOTLY_DARK, height=300, margin=dict(t=20,b=20,l=10,r=10))
    return fig

def fig_brand_ambiguity():
    if not DATA_LOADED: return go.Figure()
    token_brand = defaultdict(lambda: {'BRAND':0,'O':0})
    for r in train_orig:
        for tok, lbl in zip(r['tokens'], r['labels']):
            t = tok.lower()
            if 'BRAND' in lbl: token_brand[t]['BRAND'] += 1
            elif lbl == 'O':   token_brand[t]['O'] += 1
    inconsistent = sorted(
        [(tok, c) for tok, c in token_brand.items() if c['BRAND']>5 and c['O']>5],
        key=lambda x: x[1]['BRAND'], reverse=True
    )[:10]
    toks = [t for t,_ in inconsistent]
    brand_c = [c['BRAND'] for _,c in inconsistent]
    o_c     = [c['O']     for _,c in inconsistent]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Tagged as BRAND', x=toks, y=brand_c, marker_color=COLORS['blue']))
    fig.add_trace(go.Bar(name='Tagged as O',     x=toks, y=o_c,     marker_color=P['border']))
    fig.update_layout(**PLOTLY_DARK, barmode='stack', height=280,
                      yaxis=dict(gridcolor=P['border'], title='Count'),
                      legend=dict(bgcolor=P['bg'], bordercolor=P['border']))
    return fig

def fig_resolution_variability():
    if not DATA_LOADED: return go.Figure()
    res_vals = Counter(
        val.lower() for r in train_orig
        for etype, val, _, _ in extract_entities(r['tokens'], r['labels'])
        if etype == 'RESOLUTION'
    )
    top15 = res_vals.most_common(15)
    vals = [v for v,_ in top15]; cnts = [c for _,c in top15]
    fig = go.Figure(go.Bar(
        x=vals, y=cnts, marker_color=COLORS['pink'],
        text=cnts, textposition='outside', textfont=dict(color=P['text']),
    ))
    fig.update_layout(**PLOTLY_DARK, height=280, showlegend=False,
                      yaxis=dict(gridcolor=P['border'], title='Count'),
                      xaxis=dict(tickangle=30),
                      margin=dict(t=40,b=60,l=40,r=20),
                      annotations=[dict(
                          text=f'474 unique surface forms for {sum(res_vals.values())} total spans',
                          xref='paper', yref='paper', x=0.5, y=1.08,
                          showarrow=False, font=dict(color=P['muted'], size=11),
                      )])
    return fig

def fig_cleaning_matrix():
    cm = np.array([[0,309,4208,21,40],[452,0,148,18,3],[201,270,0,15,10],[78,64,50,0,1],[2,3,0,0,0]])
    fig = go.Figure(go.Heatmap(
        z=cm, x=ENTITY_CLASSES, y=ENTITY_CLASSES,
        colorscale='YlOrRd', text=cm, texttemplate='%{text}', textfont=dict(size=11),
    ))
    fig.update_layout(**PLOTLY_DARK, height=320,
                      xaxis_title='Corrected label', yaxis_title='Original label',
                      margin=dict(t=20,b=60,l=100,r=20))
    return fig

def fig_brand_comparison():
    if not DATA_LOADED: return go.Figure()
    orig_top  = top_tokens_for_class(train_orig,  'BRAND', 8)
    clean_top = top_tokens_for_class(train_clean, 'BRAND', 8)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Original', x=[v for v,_ in orig_top],  y=[c for _,c in orig_top],  marker_color=COLORS['blue']))
    fig.add_trace(go.Bar(name='Cleaned',  x=[v for v,_ in clean_top], y=[c for _,c in clean_top], marker_color=COLORS['orange']))
    fig.update_layout(**PLOTLY_DARK, barmode='group', height=280,
                      yaxis=dict(gridcolor=P['border'], title='Count'),
                      legend=dict(bgcolor=P['bg'], bordercolor=P['border']))
    return fig

def fig_processor_comparison():
    if not DATA_LOADED: return go.Figure()
    orig_top  = top_tokens_for_class(train_orig,  'PROCESSOR', 8)
    clean_top = top_tokens_for_class(train_clean, 'PROCESSOR', 8)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Original', x=[v for v,_ in orig_top],  y=[c for _,c in orig_top],  marker_color=COLORS['blue']))
    fig.add_trace(go.Bar(name='Cleaned',  x=[v for v,_ in clean_top], y=[c for _,c in clean_top], marker_color=COLORS['green']))
    fig.update_layout(**PLOTLY_DARK, barmode='group', height=280,
                      yaxis=dict(gridcolor=P['border'], title='Count'),
                      legend=dict(bgcolor=P['bg'], bordercolor=P['border']))
    return fig

def fig_perclass_f1(selected):
    fig = go.Figure()
    for i, cls in enumerate(ENTITY_CLASSES):
        fig.add_trace(go.Bar(
            name=cls,
            x=[k.replace('\n',' ') for k in selected],
            y=[RESULTS[k][cls] for k in selected if k in RESULTS],
            marker_color=list(ENTITY_COLORS.values())[i],
        ))
    fig.update_layout(**PLOTLY_DARK, barmode='group', height=320,
                      yaxis=dict(range=[0,1.05], gridcolor=P['border'], title='F1'),
                      xaxis=dict(gridcolor=P['border']),
                      legend=dict(bgcolor=P['bg'], bordercolor=P['border']))
    return fig

def fig_micro_f1_summary(model_prefix):
    keys   = [k for k in RESULTS if model_prefix in k]
    labels = [k.replace('\n',' ') for k in keys]
    values = [RESULTS[k]['micro_f1'] for k in keys]
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors[:len(values)],
        text=[f'{v:.4f}' for v in values],
        textposition='outside', textfont=dict(color=P['text'], size=11),
    ))
    fig.update_layout(
        paper_bgcolor=P['card'], plot_bgcolor=P['card'],
        font=dict(color=P['text'], family='monospace'),
        height=280, showlegend=False,
        yaxis=dict(range=[0,0.88], gridcolor=P['border'], title='Micro F1'),
        xaxis=dict(gridcolor=P['border']),
        margin=dict(t=40, b=20, l=40, r=20),
    )
    return fig

def fig_training_curves(model_prefix):
    hist_orig  = HISTORY[f'{model_prefix}_orig']
    hist_clean = HISTORY[f'{model_prefix}_clean']
    fig_mpl, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=P['card'])
    for ax, hist, title in zip(axes,
        [hist_orig, hist_clean],
        ['Experiment 1 — Original labels', 'Experiment 2 — Cleaned labels']):
        ax.set_facecolor(P['card'])
        ax2 = ax.twinx()
        ax.plot(hist['train_loss'], color=COLORS['blue'],   lw=1.8, label='Train loss')
        ax2.plot(hist['val_f1'],   color=COLORS['orange'],  lw=1.8, ls='--', label='Val F1')
        ax.set_xlabel('Epoch', color=P['text'])
        ax.set_ylabel('Loss',  color=P['text'])
        ax2.set_ylabel('Val micro F1', color=P['text'])
        ax.set_title(title, color=P['text'], fontweight='bold')
        ax.tick_params(colors=P['text']); ax2.tick_params(colors=P['text'])
        for spine in ax.spines.values(): spine.set_edgecolor(P['border'])
        l1,n1 = ax.get_legend_handles_labels(); l2,n2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, n1+n2, loc='upper right', fontsize=8, facecolor=P['bg'], labelcolor=P['text'])
    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=P['card'])
    plt.close(fig_mpl); buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

def fig_confusion_matrix(model_key):
    cm = CM_DATA[model_key]
    fig_mpl, ax = plt.subplots(figsize=(7,5), facecolor=P['card'])
    ax.set_facecolor(P['card'])
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax).ax.tick_params(colors=P['text'])
    ax.set_xticks(range(len(CM_COLS))); ax.set_xticklabels(CM_COLS, rotation=30, ha='right', fontsize=9, color=P['text'])
    ax.set_yticks(range(len(ENTITY_CLASSES))); ax.set_yticklabels(ENTITY_CLASSES, fontsize=9, color=P['text'])
    ax.set_xlabel('Predicted', color=P['text']); ax.set_ylabel('Gold', color=P['text'])
    for spine in ax.spines.values(): spine.set_edgecolor(P['border'])
    for i in range(len(ENTITY_CLASSES)):
        for j in range(len(CM_COLS)):
            if cm[i,j]>0:
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                        color='white' if cm[i,j]>cm.max()*0.5 else P['text'])
    plt.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=P['card'])
    plt.close(fig_mpl); buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()
# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'MAVE Laptops NER'

# ── UI helpers ─────────────────────────────────────────────────────────────────
def card(children, style=None):
    s = {'backgroundColor': P['card'], 'border': f'1px solid {P["border"]}',
         'borderRadius': '8px', 'padding': '20px 24px', 'marginBottom': '16px'}
    if style: s.update(style)
    return html.Div(children, style=s)

def section_title(text):
    return html.H3(text, style={
        'color': P['accent'], 'fontFamily': 'monospace', 'fontSize': '11px',
        'letterSpacing': '0.12em', 'textTransform': 'uppercase',
        'marginBottom': '14px', 'marginTop': '0',
        'borderBottom': f'1px solid {P["border"]}', 'paddingBottom': '8px',
    })

def kpi(label, value, color=None):
    return html.Div([
        html.Div(value, style={'fontSize': '26px', 'fontWeight': 'bold',
                               'color': color or P['accent'], 'fontFamily': 'monospace'}),
        html.Div(label, style={'fontSize': '10px', 'color': P['muted'],
                               'fontFamily': 'monospace', 'marginTop': '4px',
                               'letterSpacing': '0.08em', 'textTransform': 'uppercase'}),
    ], style={'backgroundColor': P['card'], 'border': f'1px solid {P["border"]}',
              'borderRadius': '8px', 'padding': '14px 18px', 'flex': '1', 'minWidth': '130px'})

def param_table(params):
    return html.Table([html.Tbody([
        html.Tr([
            html.Td(k, style={'color': P['muted'], 'padding': '5px 12px 5px 0',
                              'fontFamily': 'monospace', 'fontSize': '12px', 'whiteSpace': 'nowrap'}),
            html.Td(v, style={'color': P['text'], 'padding': '5px 0',
                              'fontFamily': 'monospace', 'fontSize': '12px', 'fontWeight': 'bold'}),
        ]) for k, v in params
    ])], style={'width': '100%', 'borderCollapse': 'collapse'})

def entity_badge(text, entity_type):
    color = ENTITY_COLORS.get(entity_type, 'transparent')
    children = [html.Span(text)]
    if entity_type != 'O':
        children.append(html.Sup(entity_type, style={'fontSize': '9px', 'marginLeft': '2px'}))
    return html.Span(children, style={
        'backgroundColor': color,
        'color': '#000' if entity_type != 'O' else P['text'],
        'padding': '2px 6px', 'borderRadius': '4px',
        'marginRight': '4px', 'marginBottom': '4px',
        'display': 'inline-block', 'fontFamily': 'monospace', 'fontSize': '13px',
        'opacity': '0.9' if entity_type != 'O' else '1',
    })

def render_annotated(tokens, labels):
    spans = []
    for tok, lab in zip(tokens, labels):
        etype = lab.split('-')[1] if '-' in lab else 'O'
        spans.append(entity_badge(tok, etype))
    return html.Div(spans, style={'lineHeight': '2.2', 'padding': '10px'})

def entity_legend():
    return html.Div([
        html.Span(cls, style={
            'backgroundColor': ENTITY_COLORS[cls], 'color': '#000',
            'padding': '2px 8px', 'borderRadius': '4px', 'fontSize': '11px',
            'fontFamily': 'monospace', 'fontWeight': 'bold', 'marginRight': '8px',
        }) for cls in ENTITY_CLASSES
    ], style={'marginBottom': '12px'})

def sidebar(items, active, prefix):
    """items = list of (key, label) tuples"""
    return html.Div([
        html.Div([
            html.Button(label, id=f'sb-{prefix}-{key}', n_clicks=0, style={
                'display': 'block', 'width': '100%', 'textAlign': 'left',
                'padding': '10px 16px', 'border': 'none', 'cursor': 'pointer',
                'fontFamily': 'monospace', 'fontSize': '12px',
                'backgroundColor': P['active'] if key == active else 'transparent',
                'color': P['accent'] if key == active else P['muted'],
                'borderLeft': f'3px solid {P["accent"]}' if key == active else f'3px solid transparent',
                'borderRadius': '0',
            }) for key, label in items
        ]),
    ], style={
        'backgroundColor': P['sidebar'],
        'border': f'1px solid {P["border"]}',
        'borderRadius': '8px',
        'width': '180px',
        'minWidth': '180px',
        'padding': '8px 0',
        'position': 'sticky',
        'top': '20px',
        'alignSelf': 'flex-start',
    })

# ── Tab content builders ───────────────────────────────────────────────────────
def tab_dataset(section='overview'):
    items = [('overview', '📊 Overview'), ('noise', '🔍 Annotation Noise')]
    orig_counts = span_counts(train_orig) if DATA_LOADED else {}
    total_seqs  = len(train_orig)
    o_ratio = sum(1 for r in train_orig for l in r['labels'] if l=='O') / max(sum(len(r['labels']) for r in train_orig),1) if DATA_LOADED else 0

    if section == 'overview':
        content = html.Div([
            html.Div([
                kpi('Sequences', f'{total_seqs:,}'),
                kpi('Entity classes', '5'),
                kpi('O token ratio', f'{o_ratio:.1%}'),
                kpi('BRAND spans', f'{orig_counts.get("BRAND",0):,}', ENTITY_COLORS['BRAND']),
                kpi('BATTERY spans', f'{orig_counts.get("BATTERY",0):,}', ENTITY_COLORS['BATTERY']),
            ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'16px'}),
            card([section_title('Entity Span Count per Class'),
                  dcc.Graph(figure=fig_label_dist(), config={'displayModeBar':False})]),
            card([section_title('Entity Co-occurrence Matrix'),
                  dcc.Graph(figure=fig_cooccurrence(), config={'displayModeBar':False}),
                  html.P('Each cell counts sequences where both classes co-occur.',
                         style={'color':P['muted'],'fontSize':'11px','marginTop':'8px'})]),
        ])
    else:
        content = html.Div([
            card([section_title('BRAND Annotation Inconsistency'),
                  dcc.Graph(figure=fig_brand_ambiguity(), config={'displayModeBar':False}),
                  html.P('Same token tagged as BRAND in some sequences and O in others — direct evidence of systematic mislabelling.',
                         style={'color':P['muted'],'fontSize':'11px','marginTop':'8px'})]),
            card([section_title('RESOLUTION Surface Form Variability'),
                  dcc.Graph(figure=fig_resolution_variability(), config={'displayModeBar':False}),
                  html.P('474 distinct surface forms for the same semantic concept — primary driver of low RESOLUTION F1.',
                         style={'color':P['muted'],'fontSize':'11px','marginTop':'8px'})]),
        ])

    return html.Div([sidebar(items, section, 'dataset'), html.Div(content, style={'flex':'1','minWidth':'0'})],
                   style={'display':'flex','gap':'20px','alignItems':'flex-start'})


def tab_cleaning(section='overview'):
    items = [('overview', '📋 Pipeline'), ('impact', '📈 Impact')]

    if section == 'overview':
        content = html.Div([
            html.Div([
                kpi('Spans corrected', '7,273', COLORS['orange']),
                kpi('Correction rate', '47.7%', COLORS['orange']),
                kpi('By rule', '26.6%'),
                kpi('By LLM', '73.4%'),
                kpi('API failures', '1', COLORS['red']),
            ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'16px'}),
            card([section_title('Span Correction Matrix — Train Split'),
                  dcc.Graph(figure=fig_cleaning_matrix(), config={'displayModeBar':False}),
                  html.P('Dominant pattern: BRAND → PROCESSOR (4,208 corrections). The original benchmark rewards this noise.',
                         style={'color':P['muted'],'fontSize':'11px','marginTop':'8px'})]),
        ])
    else:
        content = html.Div([
            card([section_title('BRAND Token Distribution: Before vs After'),
                  dcc.Graph(figure=fig_brand_comparison(), config={'displayModeBar':False})]),
            card([section_title('PROCESSOR Token Distribution: Before vs After'),
                  dcc.Graph(figure=fig_processor_comparison(), config={'displayModeBar':False})]),
        ])

    return html.Div([sidebar(items, section, 'cleaning'), html.Div(content, style={'flex':'1','minWidth':'0'})],
                   style={'display':'flex','gap':'20px','alignItems':'flex-start'})


def tab_bilstm(section='arch'):
    items = [('arch', '🏗 Architecture'), ('results', '📈 Results')]

    if section == 'arch':
        content = html.Div([
            html.Div([
                html.Div([
                    card([section_title('Model Parameters'), param_table(BILSTM_PARAMS)]),
                ], style={'flex':'1','minWidth':'260px'}),
                html.Div([
                    card([
                        section_title('Architecture: GloVe + Char-CNN + BiLSTM + CRF'),
                        html.P([html.Strong('GloVe 300d ', style={'color':P['accent']}),
                                'maps each token to a 300-dim pre-trained vector encoding semantic similarity. Coverage on MAVE: 15.5% — low due to technical alphanumeric tokens.'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('Char-CNN ', style={'color':P['accent']}),
                                'produces a 100-dim character-level representation per token via convolutional filters (kernel sizes 2,3) + max-pooling. Handles unseen and low-frequency tokens independently of GloVe coverage.'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('BiLSTM (2 layers, hidden=256) ', style={'color':P['accent']}),
                                'processes the concatenated 400-dim input bidirectionally, producing context-sensitive hidden states h_t = [→h_t; ←h_t] ∈ ℝ⁵¹².'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('CRF output layer ', style={'color':P['accent']}),
                                'models label transition probabilities, enforcing globally consistent BIO sequences via Viterbi decoding. Prevents invalid transitions (e.g. I-BRAND after O).'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('Tokenization: ', style={'color':P['accent']}),
                                'the dataset is pre-tokenized (MAVE JSONL format). Each token is mapped to a word index (word2id built from training data) and a character index sequence. No subword splitting — each MAVE token is treated as atomic.'],
                               style={'color':P['text'],'fontSize':'12px'}),
                    ]),
                ], style={'flex':'2','minWidth':'300px'}),
            ], style={'display':'flex','gap':'16px','flexWrap':'wrap'}),
        ])
    else:
        content = html.Div([
            html.Div([
                kpi('Orig→Orig', '0.7346', COLORS['blue']),
                kpi('Clean→Orig', '0.5293', COLORS['orange']),
                kpi('Clean→Clean', '0.7518', COLORS['green']),
            ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'16px'}),
            card([section_title('Micro F1 Summary'),
                  dcc.Graph(figure=fig_micro_f1_summary('BiLSTM'), config={'displayModeBar':False})]),
            card([section_title('Per-Class F1 Comparison'),
                  dcc.Graph(figure=fig_perclass_f1(['BiLSTM\nOrig→Orig','BiLSTM\nClean→Orig','BiLSTM\nClean→Clean']),
                            config={'displayModeBar':False})]),
            card([section_title('Training Curves'),
                  html.Img(src=fig_training_curves('bilstm'), style={'width':'100%','borderRadius':'4px'})]),
            html.Div([
                html.Div([card([section_title('Confusion Matrix — Experiment 1'),
                                html.Img(src=fig_confusion_matrix('bilstm_orig'), style={'width':'100%'})])],
                         style={'flex':'1','minWidth':'300px'}),
                html.Div([card([section_title('Confusion Matrix — Experiment 2 (Cleaned)'),
                                html.Img(src=fig_confusion_matrix('bilstm_clean'), style={'width':'100%'})])],
                         style={'flex':'1','minWidth':'300px'}),
            ], style={'display':'flex','gap':'16px','flexWrap':'wrap'}),
        ])

    return html.Div([sidebar(items, section, 'bilstm'), html.Div(content, style={'flex':'1','minWidth':'0'})],
                   style={'display':'flex','gap':'20px','alignItems':'flex-start'})


def tab_deberta(section='arch'):
    items = [('arch', '🏗 Architecture'), ('results', '📈 Results')]

    if section == 'arch':
        content = html.Div([
            html.Div([
                html.Div([
                    card([section_title('Model Parameters'), param_table(DEBERTA_PARAMS)]),
                ], style={'flex':'1','minWidth':'260px'}),
                html.Div([
                    card([
                        section_title('Architecture: DeBERTa-v3-base'),
                        html.P([html.Strong('DeBERTa-v3-base ', style={'color':P['accent']}),
                                'uses disentangled attention — content and relative position are encoded separately and combined via four interaction terms. Particularly effective for technical entities where position carries semantic information (e.g. "15.6-inch", "i7-1165G7").'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('ELECTRA-style pre-training ', style={'color':P['accent']}),
                                '(DeBERTa-v3) replaces masked language modelling with replaced token detection — a generator produces substitutions and the discriminator classifies each token as real or replaced, providing a denser training signal than standard MLM.'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('SentencePiece tokenization ', style={'color':P['accent']}),
                                'splits each pre-tokenized MAVE token into subword units (e.g. i7-1165G7 → [i7, -, 1165, G7]). BIO labels are aligned to subwords: the first subword of each token receives the gold label; subsequent subwords are assigned -100 and ignored in loss and evaluation.'],
                               style={'color':P['text'],'fontSize':'12px','marginBottom':'10px'}),
                        html.P([html.Strong('Fine-tuning setup: ', style={'color':P['accent']}),
                                'AutoModelForTokenClassification with a linear classification head over 11 BIO labels. The classifier head is re-initialized with Xavier uniform weights before each experiment. .float() cast applied to prevent float16 NaN gradients on T4 GPU.'],
                               style={'color':P['text'],'fontSize':'12px'}),
                    ]),
                ], style={'flex':'2','minWidth':'300px'}),
            ], style={'display':'flex','gap':'16px','flexWrap':'wrap'}),
        ])
    else:
        content = html.Div([
            html.Div([
                kpi('Orig→Orig', '0.7456', COLORS['blue']),
                kpi('Clean→Orig', '0.5346', COLORS['orange']),
                kpi('Clean→Clean', '0.7689', COLORS['green']),
            ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'16px'}),
            card([section_title('Micro F1 Summary'),
                  dcc.Graph(figure=fig_micro_f1_summary('DeBERTa'), config={'displayModeBar':False})]),
            card([section_title('Per-Class F1 Comparison'),
                  dcc.Graph(figure=fig_perclass_f1(['DeBERTa\nOrig→Orig','DeBERTa\nClean→Orig','DeBERTa\nClean→Clean']),
                            config={'displayModeBar':False})]),
            card([section_title('Training Curves'),
                  html.Img(src=fig_training_curves('deberta'), style={'width':'100%','borderRadius':'4px'})]),
            html.Div([
                html.Div([card([section_title('Confusion Matrix — Experiment 1'),
                                html.Img(src=fig_confusion_matrix('deberta_orig'), style={'width':'100%'})])],
                         style={'flex':'1','minWidth':'300px'}),
                html.Div([card([section_title('Confusion Matrix — Experiment 2 (Cleaned)'),
                                html.Img(src=fig_confusion_matrix('deberta_clean'), style={'width':'100%'})])],
                         style={'flex':'1','minWidth':'300px'}),
            ], style={'display':'flex','gap':'16px','flexWrap':'wrap'}),
        ])

    return html.Div([sidebar(items, section, 'deberta'), html.Div(content, style={'flex':'1','minWidth':'0'})],
                   style={'display':'flex','gap':'20px','alignItems':'flex-start'})


def tab_livedemo():
    return html.Div([
        card([
            section_title('🔍 Live NER Demo'),
            html.P('Enter a laptop product title and see predictions from all 4 models.',
                   style={'color':P['muted'],'fontSize':'12px','marginBottom':'14px'}),
            html.Div([
                dcc.Input(id='demo-input', type='text',
                    placeholder='e.g. Dell XPS 15 15.6 inch FHD 1920x1080 Intel Core i7 laptop...',
                    style={'width':'80%','padding':'10px 14px','backgroundColor':P['bg'],
                           'border':f'1px solid {P["accent"]}','borderRadius':'6px',
                           'color':P['text'],'fontFamily':'monospace','fontSize':'13px','outline':'none'},
                    debounce=False),
                html.Button('▶  Predict', id='demo-btn', n_clicks=0, style={
                    'marginLeft':'10px','padding':'10px 20px',
                    'backgroundColor':P['accent'],'color':'#000',
                    'border':'none','borderRadius':'6px','cursor':'pointer',
                    'fontFamily':'monospace','fontWeight':'bold','fontSize':'13px',
                }),
            ], style={'marginBottom':'14px','display':'flex','alignItems':'center'}),
            html.Div([
                html.P('Quick examples:', style={'color':P['muted'],'fontSize':'11px','marginBottom':'8px'}),
                *[html.Button(ex, id=f'demo-ex-{i}', n_clicks=0, style={
                    'marginRight':'8px','marginBottom':'8px','padding':'4px 10px',
                    'backgroundColor':P['bg'],'color':P['text'],
                    'border':f'1px solid {P["border"]}','borderRadius':'4px',
                    'cursor':'pointer','fontSize':'11px','fontFamily':'monospace',
                }) for i, ex in enumerate(EXAMPLE_TEXTS)],
            ], style={'marginBottom':'16px'}),
            entity_legend(),
            html.Div(id='demo-output'),
        ]),
    ])
# ── App layout ─────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1('MAVE Laptops NER', style={
            'fontFamily': 'monospace', 'color': P['accent'],
            'letterSpacing': '0.15em', 'marginBottom': '4px', 'fontSize': '20px',
        }),
        html.P('Named Entity Recognition — Text Mining, Prof. Andrea Belli', style={
            'color': P['muted'], 'fontFamily': 'monospace', 'fontSize': '12px',
        }),
        html.Div([
            html.Span(cls, style={
                'backgroundColor': ENTITY_COLORS[cls], 'color': '#000',
                'padding': '2px 10px', 'borderRadius': '4px',
                'marginRight': '8px', 'fontSize': '11px',
                'fontFamily': 'monospace', 'fontWeight': 'bold',
            }) for cls in ENTITY_CLASSES
        ], style={'marginTop': '10px'}),
    ], style={
        'backgroundColor': P['card'], 'border': f'1px solid {P["border"]}',
        'borderRadius': '8px', 'padding': '20px 28px', 'marginBottom': '20px',
    }),

    # Tabs
    dcc.Tabs(id='tabs', value='tab-dataset', children=[
        dcc.Tab(label='📊 Dataset',       value='tab-dataset',
                style={'fontFamily':'monospace','backgroundColor':P['bg'],'color':P['muted'],'border':'none'},
                selected_style={'fontFamily':'monospace','backgroundColor':P['card'],'color':P['accent'],'borderTop':f'2px solid {P["accent"]}','border':'none'}),
        dcc.Tab(label='🧹 Label Cleaning', value='tab-cleaning',
                style={'fontFamily':'monospace','backgroundColor':P['bg'],'color':P['muted'],'border':'none'},
                selected_style={'fontFamily':'monospace','backgroundColor':P['card'],'color':P['accent'],'borderTop':f'2px solid {P["accent"]}','border':'none'}),
        dcc.Tab(label='🧠 BiLSTM + CRF',  value='tab-bilstm',
                style={'fontFamily':'monospace','backgroundColor':P['bg'],'color':P['muted'],'border':'none'},
                selected_style={'fontFamily':'monospace','backgroundColor':P['card'],'color':P['accent'],'borderTop':f'2px solid {P["accent"]}','border':'none'}),
        dcc.Tab(label='🤖 DeBERTa-v3',    value='tab-deberta',
                style={'fontFamily':'monospace','backgroundColor':P['bg'],'color':P['muted'],'border':'none'},
                selected_style={'fontFamily':'monospace','backgroundColor':P['card'],'color':P['accent'],'borderTop':f'2px solid {P["accent"]}','border':'none'}),
        dcc.Tab(label='🔍 Live Demo',      value='tab-demo',
                style={'fontFamily':'monospace','backgroundColor':P['bg'],'color':P['muted'],'border':'none'},
                selected_style={'fontFamily':'monospace','backgroundColor':P['card'],'color':P['accent'],'borderTop':f'2px solid {P["accent"]}','border':'none'}),
    ], style={'backgroundColor': P['bg']}),

    # Stores for sidebar state
    dcc.Store(id='store-dataset',  data='overview'),
    dcc.Store(id='store-cleaning', data='overview'),
    dcc.Store(id='store-bilstm',   data='arch'),
    dcc.Store(id='store-deberta',  data='arch'),

    html.Div(id='tab-content', style={'marginTop': '16px'}),

], style={
    'backgroundColor': P['bg'], 'minHeight': '100vh',
    'padding': '28px', 'fontFamily': 'monospace',
})


# ── Tab router ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('store-dataset',  'data'),
    Input('store-cleaning', 'data'),
    Input('store-bilstm',   'data'),
    Input('store-deberta',  'data'),
)
def render_tab(tab, s_dataset, s_cleaning, s_bilstm, s_deberta):
    if tab == 'tab-dataset':  return tab_dataset(s_dataset or 'overview')
    if tab == 'tab-cleaning': return tab_cleaning(s_cleaning or 'overview')
    if tab == 'tab-bilstm':   return tab_bilstm(s_bilstm or 'arch')
    if tab == 'tab-deberta':  return tab_deberta(s_deberta or 'arch')
    if tab == 'tab-demo':     return tab_livedemo()
    return html.Div()


# ── Sidebar callbacks ──────────────────────────────────────────────────────────
@app.callback(Output('store-dataset', 'data'),
              Input('sb-dataset-overview', 'n_clicks'),
              Input('sb-dataset-noise',    'n_clicks'),
              prevent_initial_call=True)
def sb_dataset(n1, n2):
    triggered = dash_ctx.triggered_id
    return 'noise' if triggered == 'sb-dataset-noise' else 'overview'

@app.callback(Output('store-cleaning', 'data'),
              Input('sb-cleaning-overview', 'n_clicks'),
              Input('sb-cleaning-impact',   'n_clicks'),
              prevent_initial_call=True)
def sb_cleaning(n1, n2):
    triggered = dash_ctx.triggered_id
    return 'impact' if triggered == 'sb-cleaning-impact' else 'overview'

@app.callback(Output('store-bilstm', 'data'),
              Input('sb-bilstm-arch',    'n_clicks'),
              Input('sb-bilstm-results', 'n_clicks'),
              prevent_initial_call=True)
def sb_bilstm(n1, n2):
    triggered = dash_ctx.triggered_id
    return 'results' if triggered == 'sb-bilstm-results' else 'arch'

@app.callback(Output('store-deberta', 'data'),
              Input('sb-deberta-arch',    'n_clicks'),
              Input('sb-deberta-results', 'n_clicks'),
              prevent_initial_call=True)
def sb_deberta(n1, n2):
    triggered = dash_ctx.triggered_id
    return 'results' if triggered == 'sb-deberta-results' else 'arch'


# ── Live demo callbacks ────────────────────────────────────────────────────────
MODELS_DEMO = {
    'BiLSTM (Original)':  'bilstm_orig',
    'BiLSTM (Cleaned)':   'bilstm_clean',
    'DeBERTa (Original)': 'deberta_orig',
    'DeBERTa (Cleaned)':  'deberta_clean',
}

@app.callback(
    Output('demo-input', 'value'),
    [Input(f'demo-ex-{i}', 'n_clicks') for i in range(5)],
    prevent_initial_call=True,
)
def fill_example(*args):
    triggered = dash_ctx.triggered_id
    if not triggered: return ''
    idx = int(triggered.split('-')[-1])
    return EXAMPLE_TEXTS[idx]

@app.callback(
    Output('demo-output', 'children'),
    Input('demo-btn', 'n_clicks'),
    State('demo-input', 'value'),
    prevent_initial_call=True,
)
def run_demo(n_clicks, text):
    if not text or not text.strip():
        return html.P('Please enter a product title.', style={'color': P['muted']})
    tokens = text.strip().split()
    results_divs = []
    for model_label, model_key in MODELS_DEMO.items():
        labels   = predict(model_key, tokens)
        entities = extract_entities(tokens, labels)
        summary  = ', '.join(f'{et}: "{v}"' for et,v,_,_ in entities) if entities else 'No entities detected'
        results_divs.append(html.Div([
            html.H6(model_label, style={
                'color': P['accent'], 'fontFamily': 'monospace',
                'marginBottom': '8px', 'fontSize': '13px',
            }),
            render_annotated(tokens, labels),
            html.P(f'Entities: {summary}', style={
                'color': P['muted'], 'fontSize': '11px',
                'marginTop': '6px', 'fontFamily': 'monospace',
            }),
        ], style={
            'backgroundColor': P['bg'], 'border': f'1px solid {P["border"]}',
            'borderRadius': '6px', 'padding': '14px', 'marginBottom': '12px',
        }))
    return html.Div(results_divs)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Starting MAVE Laptops NER Dashboard...')
    print('Open: http://127.0.0.1:8050')
    app.run(debug=False, port=8050)
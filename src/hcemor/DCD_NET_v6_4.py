"""DCANet: Deep Cognitive Adaptive Network (model of the HCEMOR framework).

Model implementation accompanying the paper
"Enhancing Understanding of Doctor-Patient Relationships through Knowledge
Sharing in Online Communities: A Deep Learning Framework with Open Set
Recognition" (INFORMS Journal on Computing).

This module implements the DCANet semantic-routing backbone together with the
MCLUFI token-level probing heads:

- CDSE1 (Cognitively Driven Semantic Embedding for parent classes):
  ``M`` parent-class cognitive blocks (``l1expert``) plus one background
  cognitive block (``backexpert``), routed by a variance-gated affinity
  router with learnable parameters ``alpha1`` / ``theta1`` (Eqs. 1-4).
- CDSE2 (Cognitively Driven Semantic Embedding for subclasses):
  ``S`` subclass cognitive blocks (``l2expert``) plus a background block,
  routed by a subclass router (``alpha2`` / ``theta2``). Expert-defined
  hierarchical knowledge (parent-to-subclass mapping matrix ``c2m``, HCKA)
  is injected into the subclass routing weights (Eqs. 6-7), and a
  reconstruction router (``router_filt`` with learnable ``filtalpha``)
  dynamically scales down known-class semantics to preserve semantic space
  for unknown factors (Eqs. 8-10).
- MCLUFI: ``P`` (= ``granularity``) linear probe networks (``extraquery``)
  operating on the reconstructed semantics. They support three contrastive
  objectives computed here: semantic reconstruction loss (``disloss_1``,
  Eq. 11), probe diversity loss (``varloss``, Eqs. 12-13), and novel-class
  exploration loss (``disloss_2``, Eqs. 14-15). The returned ``xlogit``
  (LogSumExp-smoothed maximum probe activation) is the statistic used for
  Weibull-based open-set detection after training (Eq. 20).

All textual inputs are fixed RoBERTa embeddings (768-d, last hidden layer).
The embedding encoder is frozen, so no RoBERTa parameters appear in this
module. The known-class classification head ``l2out`` is trained with the
binary cross-entropy loss (Eq. 5).

Note: GPU selection is intentionally left to the training/evaluation script
(``torch.cuda.set_device``) so that this module stays device-agnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Computation device. 'cuda' without an index refers to the *current* CUDA
# device, which the training script can change via torch.cuda.set_device().
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Bottleneck width of every cognitive block / expert module (768 -> 256 -> 768).
lowerdim = 256


def router_filt(a, ncls, alpha):
    """Reconstruction router of CDSE2 (semantic filtering, Eqs. 8-10).

    Given the subclass routing logits ``a`` (shape ``(S+1, L, B)``: S subclass
    affinities + 1 background affinity), the router

    1. normalizes the raw affinities over tokens (``tempatt``) and splits them
       into subclass part ``a1`` and background part ``a2``;
    2. re-normalizes the *unnormalized* affinities together with a learnable
       background affinity ``alpha`` to obtain the routing weights ``attb``;
    3. reconstructs a filtered weight vector in which the weights of the
       known-class cognitive networks are dynamically scaled down, while the
       background knowledge carried by each token is preserved.

    Args:
        a: subclass routing logits, shape ``(S+1, seq_len, batch_size)``.
        ncls: number of known subclasses ``S``.
        alpha: learnable background affinity (``self.filtalpha``).

    Returns:
        Filtered routing weights of shape ``(S+2, seq_len, batch_size)``
        (S known-class weights + reconstructed background weight + raw
        background weight), consumed by the MCLUFI probing stage.
    """
    tempatt = torch.softmax(a, dim=0)
    a1 = tempatt[:ncls, :, :]                     # subclass-level affinities
    a2 = tempatt[-1, :, :].unsqueeze(0)           # background affinity
    b1 = a[:ncls, :, :]
    b2 = torch.ones_like(a2) * alpha              # learnable background affinity
    attb = torch.softmax(torch.cat((b1, b2)), dim=0)
    att1 = attb[-1, :, :].repeat(ncls, 1, 1)      # background weight per subclass
    att2 = 1 - attb[-1, :, :]                     # complementary known weight
    att3 = torch.sum(a1, dim=0)                   # aggregated subclass semantics
    attention1 = att1 * a1                        # scaled-down known-class weights
    attention2 = att2 * att3
    attention2 = attention2.unsqueeze(0)          # reconstructed background weight

    att = torch.cat((attention1, attention2, a2), dim=0)
    return att


def attadd(a, ncls, c2m):
    """Inject expert-defined hierarchical knowledge into routing weights (HCKA).

    Implements the weight coupling between the two hierarchy levels: the
    routing weight assigned by each CDSE1 (parent) cognitive network is
    propagated to the corresponding CDSE2 (subclass) networks through the
    expert-defined parent-to-subclass mapping matrix ``c2m`` (Eq. 6-7). The
    matrix entry (m, s) is 1 iff subclass s belongs to parent class m; the
    paper's background-coupling entry (M+1, S+1) = 1 is realized by appending
    the CDSE1 background (contextual) weight as an extra row of the output.

    Args:
        a: CDSE1 routing weights, shape ``(M+1, seq_len, batch_size)``.
        ncls: number of parent classes ``M``.
        c2m: expert-defined parent-to-subclass mapping matrix, shape ``(S, M)``.

    Returns:
        Hierarchy-aligned weights of shape ``(S+1, seq_len, batch_size)``
        (S subclass rows + the coupled background row).
    """
    a_ = a[:ncls, :, :]
    a1 = torch.einsum('ij,jkl->ikl', c2m, a_)
    a2 = a[-1, :, :].unsqueeze(0)
    out = torch.cat((a1, a2), dim=0)
    return out


def cossim2(x1, x2, dim):
    """L2 distance between two embedding tensors (used by the reconstruction loss).

    Note: despite the historical name, this helper computes the Euclidean
    distance (not a cosine similarity); the reconstruction loss (Eq. 11)
    compares distances of known-only vs. potentially-unknown inputs to the
    background semantics, so smaller values mean closer semantics.

    Args:
        x1, x2: tensors with identical shape.
        dim: unused; kept for interface compatibility.

    Returns:
        Element-wise L2 distance over the last dimension.
    """
    diff_squared = torch.sum((x1 - x2) ** 2, dim=-1)
    diff_squared = torch.sqrt(diff_squared)
    return diff_squared


class DCDNet(nn.Module):
    """Deep Cognitive Adaptive Network (DCANet) with MCLUFI probing heads.

    Args:
        semantic_main: parent-class (level-1) definition embeddings,
            shape ``(M, def_len, 768)``; expert-assisted generated data.
        semantic_sub: subclass (level-2) definition embeddings,
            shape ``(S, def_len, 768)``; expert-assisted generated data.
        sem_of_nan: background semantics without specific factors, stored in
            the same orientation as the input embeddings ``(B, def_len, 768)``
            (it is permuted to sequence-first inside ``forward``).
        beta: temperature controlling how strongly the CDSE1 routing weights
            influence the CDSE2 subclass router (hierarchy coupling degree).
        c2m: expert-defined parent-to-subclass mapping matrix (HCKA),
            shape ``(S, M)``.
        granularity: number of MCLUFI probe networks ``P`` (best P = 5).
    """

    def __init__(self, semantic_main, semantic_sub, sem_of_nan, beta, c2m, granularity):
        super(DCDNet, self).__init__()
        # Expert-defined class prototypes (frozen inputs, no gradient).
        self.semantic_main = semantic_main.to(device)
        self.semantic_sub = semantic_sub.to(device)
        self.sem_of_nan = sem_of_nan.to(device)
        self.num_l1classes = semantic_main.shape[0]  # num of parent class
        self.num_l2classes = semantic_sub.shape[0]   # num of child class
        self.embeddim = 768  # RoBERTa embedding dimension
        self.max_len = 256

        # Learnable router parameters:
        # alpha1/theta1 -> CDSE1 parent router (affinity floor / variance gate),
        # alpha2/theta2 -> CDSE2 subclass router,
        # filtalpha     -> reconstruction router (background affinity).
        self.alpha1 = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=True)
        self.theta1 = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=True)
        self.theta2 = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=True)
        self.tem = torch.tensor([beta], device=device)  # hierarchy coupling temperature
        self.heads = 3
        self.filtalpha = nn.Parameter(torch.tensor([1.0], device=device), requires_grad=True)
        self.head_dim = self.embeddim // self.heads
        self.c2m = c2m.float().to(device)
        self.relu = nn.ReLU()
        self.tau = torch.tensor([0.0001], device=device)
        self.gra = granularity  # P: number of MCLUFI probe networks

        # Semantic fusion encoders (TransformerEncoder, 1 layer each).
        # globalencoder1 is kept for checkpoint compatibility with previously
        # released weights; the current architecture uses globalencoder2 as the
        # fusion encoder between CDSE1 and CDSE2.
        Tfencoder_global1 = nn.TransformerEncoderLayer(d_model=self.embeddim, nhead=3)
        Tfencoder_global2 = nn.TransformerEncoderLayer(d_model=self.embeddim, nhead=3)
        Tfencoder_select = nn.TransformerEncoderLayer(d_model=self.embeddim, nhead=3)
        Tfencoder_filt = nn.TransformerEncoderLayer(d_model=self.embeddim, nhead=3)

        self.globalencoder1 = nn.TransformerEncoder(Tfencoder_global1, num_layers=1)
        self.globalencoder2 = nn.TransformerEncoder(Tfencoder_global2, num_layers=1)
        self.selectencoder = nn.TransformerEncoder(Tfencoder_select, num_layers=1)
        self.filtencoder = nn.TransformerEncoder(Tfencoder_filt, num_layers=1)

        # Shared multi-head self-attention layers (definition/input encoding).
        self.att1 = nn.MultiheadAttention(embed_dim=self.embeddim, num_heads=3)
        self.att2 = nn.MultiheadAttention(embed_dim=self.embeddim, num_heads=3)

        self.l1_1 = nn.LayerNorm(self.embeddim)
        self.l1_2 = nn.LayerNorm(self.embeddim)
        self.l2_1 = nn.LayerNorm(self.embeddim)
        self.l2_2 = nn.LayerNorm(self.embeddim)

        # CDSE1 cognitive blocks: M parent-class blocks + (via backexpert) the
        # M+1-th background block. Each block is a bottleneck MLP.
        self.l1expert = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embeddim, 256),
                          nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(256, self.embeddim)) for _ in range(self.num_l1classes + 2)
        ])
        # CDSE2 cognitive blocks: S subclass blocks + background block.
        self.l2expert = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embeddim, 256),
                          nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(256, self.embeddim)) for _ in range(self.num_l2classes + 2)
        ])
        # Shared background cognitive block (the (M+1)-th / (S+1)-th block).
        self.backexpert = nn.Sequential(nn.Linear(self.embeddim, 256),
                                        nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(256, self.embeddim))
        # Reserved for ablation on the reconstruction router; kept for
        # checkpoint compatibility (no effect in the current forward pass).
        self.filtexpert = nn.Sequential(nn.Linear(self.embeddim, 256),
                                        nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(256, self.embeddim))

        # Shared Q/K/V projections of the prototype-guided attention.
        self.query1 = nn.Linear(self.embeddim, 768, bias=False)
        self.keys1 = nn.Linear(self.embeddim, 768, bias=False)
        self.value1 = nn.Linear(self.embeddim, 768, bias=False)

        # Linear mapping producing the synthesized semantics of the
        # reconstruction component of MCLUFI (Eq. 11).
        self.active = nn.Sequential(nn.Linear(self.embeddim, 256),
                                    nn.ReLU(), nn.Linear(256, self.embeddim))

        # Known-class classification heads: S binary classifiers + background.
        self.l2out = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embeddim, 768), nn.ReLU(),
                          nn.Dropout(0.2), nn.Linear(768, 1)) for _ in range(self.num_l2classes + 2)
        ])

        # MCLUFI: P linear probe networks (one scalar output each).
        self.extraquery = nn.ModuleList([
            nn.Linear(self.embeddim, 1, bias=False) for _ in range(self.gra)
        ])

    def forward(self, x, x_):
        """Forward pass of DCANet + MCLUFI.

        Args:
            x: embeddings of the basic data containing known factors
                (``X``), shape ``(batch, seq_len, 768)``.
            x_: embeddings of the LLM-UFDE generated data (``X'``), which may
                contain known or unknown factors, shape ``(batch, seq_len, 768)``.

        Returns:
            l2out: known-class multi-label logits, shape ``(batch, S)``.
            aux: list ``[xlogit, disloss_1, disloss_2, varloss, dis_alpha]``
                where ``xlogit`` is the LogSumExp-smoothed maximum probe
                activation on the LLM-UFDE data (open-set detection statistic),
                and the remaining entries are the MCLUFI contrastive losses
                (Eqs. 11-15) plus the diagnostic distance ``dis_alpha``.
        """
        # ------------------------------------------------------------------
        # Class-prototype token sequences (sequence-first layout for
        # MultiheadAttention): subclass definitions, parent definitions,
        # basic data, LLM-UFDE data, and background semantics.
        # ------------------------------------------------------------------
        repl2 = self.semantic_sub.permute(1, 0, 2)  # (def_len, S, 768)
        cog_l2 = repl2

        repl1 = self.semantic_main.permute(1, 0, 2)  # (def_len, M, 768)
        cog_l1 = repl1

        x1 = x.permute(1, 0, 2)      # basic data (known factors)
        x2 = x_.permute(1, 0, 2)     # LLM-UFDE generated data
        x_nan = self.sem_of_nan.permute(1, 0, 2)  # background semantics

        # ------------------------------------------------------------------
        # Prototype-guided self-attention + residual LayerNorm over the class
        # definitions and both input sources (shared Q/K/V projections).
        # ------------------------------------------------------------------
        cog_l2_att1, _ = self.att1(self.query1(cog_l2), self.keys1(cog_l2), self.value1(cog_l2))
        cog_l2 = self.l1_1(cog_l2_att1 + cog_l2)
        cog_l1_att1, _ = self.att1(self.query1(cog_l1), self.keys1(cog_l1), self.value1(cog_l1))
        cog_l1 = self.l1_1(cog_l1_att1 + cog_l1)
        x1_att1, _ = self.att1(self.query1(x1), self.keys1(x1), self.value1(x1))
        x1 = self.l1_1(x1_att1 + x1)
        x2_att1, _ = self.att1(self.query1(x2), self.keys1(x2), self.value1(x2))
        x2 = self.l1_1(x2_att1 + x2)
        x_nan_att1, _ = self.att1(self.query1(x_nan), self.keys1(x_nan), self.value1(x_nan))
        x_nan = self.l1_1(x_nan_att1 + x_nan)

        # ------------------------------------------------------------------
        # CDSE1: parent-class affinity computation (Eq. 3). The class query is
        # the semantic representation of the parent definitions averaged by
        # item; affinities are scaled dot products against the token keys of
        # every input source (scale factor sqrt(d) with d = 768).
        # ------------------------------------------------------------------
        query1 = self.query1(torch.mean(cog_l1, dim=0))  # (M, 768)
        keys1 = self.keys1(x1)                # (seq_len, batch, 768)
        keys2 = self.keys1(x2)                # (seq_len, batch, 768)
        keyscog2 = self.keys1(cog_l2)         # subclass definition keys
        keys_nan1 = self.keys1(x_nan)         # background keys

        attentionl1_1 = []
        attentionl1_2 = []
        attentionlcog_2 = []
        attentionlnan_1 = []
        x1_1 = []
        x2_1 = []
        cogl2_ = []
        x_nan_1 = []
        for i in range(query1.shape[0]):
            # M parent-class cognitive blocks applied to every input source.
            tempx1 = self.l1expert[i](x1)
            x1_1.append(tempx1)

            tempx2 = self.l1expert[i](x2)
            x2_1.append(tempx2)

            tempcog2 = self.l1expert[i](cog_l2)
            cogl2_.append(tempcog2)

            tempnan1 = self.l1expert[i](x_nan)
            x_nan_1.append(tempnan1)

            # Scaled dot-product affinity between the i-th parent query and
            # the token keys (Eq. 3).
            queryl1 = query1[i].unsqueeze(0).unsqueeze(0)
            attention1 = torch.sum(queryl1 * keys1, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionl1_1.append(attention1)

            attention2 = torch.sum(queryl1 * keys2, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionl1_2.append(attention2)

            attentioncog2 = torch.sum(queryl1 * keyscog2, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionlcog_2.append(attentioncog2)

            attentionnan = torch.sum(queryl1 * keys_nan1, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionlnan_1.append(attentionnan)

        # Append the shared background cognitive block output as the (M+1)-th
        # candidate for every input source.
        x1_1 = torch.stack(x1_1)  # (M, seq_len, batch, 768)
        x1_b = self.backexpert(x1).unsqueeze(0)
        x1_1 = torch.cat((x1_1, x1_b), dim=0)

        x2_1 = torch.stack(x2_1)
        x2_b = self.backexpert(x2).unsqueeze(0)
        x2_1 = torch.cat((x2_1, x2_b), dim=0)

        cogl2_ = torch.stack(cogl2_)
        cogl2_b = self.backexpert(cog_l2).unsqueeze(0)
        cogl2_ = torch.cat((cogl2_, cogl2_b), dim=0)

        x_nan_1 = torch.stack(x_nan_1)
        x_nan_1_b = self.backexpert(x_nan).unsqueeze(0)
        x_nan_1_ = torch.cat((x_nan_1, x_nan_1_b), dim=0)

        # Stacked affinity maps: (M, seq_len, batch).
        attentionl1_1 = torch.stack(attentionl1_1)
        attentionl1_2 = torch.stack(attentionl1_2)
        attentionlcog_2 = torch.stack(attentionlcog_2)
        attentionlnan_1 = torch.stack(attentionlnan_1)

        # ------------------------------------------------------------------
        # CDSE1 routing (Eqs. 1-4): the affinity variance across parent
        # classes acts as a token-level confidence gate
        # (sigmoid(var - theta1)); a learnable background weight alpha1
        # competes with the gated class affinities through a softmax.
        # ------------------------------------------------------------------
        # routing for the basic data x
        v1 = torch.var(attentionl1_1, dim=0, unbiased=True).unsqueeze(0)
        v1_ = v1.repeat(attentionl1_1.shape[0], 1, 1)
        vec1 = attentionl1_1 * torch.sigmoid(v1_ - self.theta1)  # (M, seq_len, batch)
        vec2 = torch.ones_like(v1) * self.alpha1                 # (1, seq_len, batch)
        att1_1 = torch.cat((vec1, vec2), dim=0)
        attention1_1 = torch.softmax(att1_1, dim=0)
        # HCKA: propagate the CDSE1 contextual weight to CDSE2 (Eqs. 6-7),
        # reused later inside the CDSE2 subclass router.
        xadd_attention = attadd(attention1_1, vec1.shape[0], self.c2m)  # (S+1, seq_len, batch)
        attention1_1 = attention1_1.unsqueeze(3)
        x1_1 = self.l1_2(x1 + torch.sum(x1_1 * attention1_1, dim=0))    # (seq_len, batch, 768)

        # routing for the LLM-UFDE data x'
        v2 = torch.var(attentionl1_2, dim=0, unbiased=True).unsqueeze(0)
        v2_ = v2.repeat(attentionl1_2.shape[0], 1, 1)
        vec1_2 = attentionl1_2 * torch.sigmoid(v2_ - self.theta1)
        vec2_2 = torch.ones_like(v2) * self.alpha1
        att2_2 = torch.cat((vec1_2, vec2_2), dim=0)
        attention1_2 = torch.softmax(att2_2, dim=0)
        attention1_2 = attention1_2.unsqueeze(3)
        x2_1 = self.l1_2(x2 + torch.sum(x2_1 * attention1_2, dim=0))    # (seq_len, batch, 768)

        # routing for the subclass definitions
        vcogl2 = torch.var(attentionlcog_2, dim=0, unbiased=True).unsqueeze(0)
        vcogl2_ = vcogl2.repeat(attentionlcog_2.shape[0], 1, 1)
        vec1_3 = attentionlcog_2 * torch.sigmoid(vcogl2_ - self.theta1)
        vec2_3 = torch.ones_like(vcogl2) * self.alpha1
        att2_3 = torch.cat((vec1_3, vec2_3), dim=0)
        attention1_3 = torch.softmax(att2_3, dim=0).unsqueeze(3)
        cogl2 = self.l1_2(cog_l2 + torch.sum(cogl2_ * attention1_3, dim=0))  # (seq_len, S, 768)

        # routing for the background semantics
        vnan_1 = torch.var(attentionlnan_1, dim=0, unbiased=True).unsqueeze(0)
        vnan_1_ = vnan_1.repeat(attentionlnan_1.shape[0], 1, 1)
        vec_nan1_1 = attentionlnan_1 * torch.sigmoid(vnan_1_ - self.theta1)
        vec_nan1_2 = torch.ones_like(vnan_1) * self.alpha1
        attnan_1 = torch.cat((vec_nan1_1, vec_nan1_2), dim=0)
        attentionnan_1 = torch.softmax(attnan_1, dim=0)
        attentionnan_1 = attentionnan_1.unsqueeze(3)
        x_nan_1_ = self.l1_2(x_nan + torch.sum(x_nan_1_ * attentionnan_1, dim=0))  # (seq_len, batch, 768)

        # ------------------------------------------------------------------
        # Semantic fusion Transformer encoder between CDSE1 and CDSE2
        # (Section 3.4.1: "The semantic fusion process is implemented through
        # a Transformer Encoder").
        # ------------------------------------------------------------------
        x1_1 = self.globalencoder2(x1_1)
        x2_1 = self.globalencoder2(x2_1)
        cogl2 = self.globalencoder2(cogl2)
        x_nan_2 = self.globalencoder2(x_nan_1_)

        # ------------------------------------------------------------------
        # CDSE2: subclass-level self-attention and affinity recomputation.
        # ------------------------------------------------------------------
        keys2_1 = self.keys1(x1_1)
        keys2_2 = self.keys1(x2_1)
        keyscogl2 = self.keys1(cogl2)
        keys_nan2 = self.keys1(x_nan_2)
        query2 = self.query1(torch.mean(cogl2, dim=0))  # (S, 768)

        x1_1_att2, _ = self.att2(self.query1(x1_1), keys2_1, self.value1(x1_1))
        x1_1 = self.l2_1(x1_1_att2 + x1_1)  # basic data
        x2_1_att2, _ = self.att2(self.query1(x2_1), keys2_2, self.value1(x2_1))
        x2_1 = self.l2_1(x2_1_att2 + x2_1)  # LLM-UFDE data
        x_nan_att2, _ = self.att2(self.query1(x_nan_2), keys_nan2, self.value1(x_nan_2))
        x_nan_2 = self.l2_1(x_nan_att2 + x_nan_2)

        keys3_1 = self.keys1(x1_1)
        keys3_2 = self.keys1(x2_1)
        keys_nan3 = self.keys1(x_nan_2)

        attentionl2_1 = []
        attentionl2_2 = []
        attentionlnan_2 = []
        x1_2 = []
        x2_2 = []
        x_nan_2_ = []
        for i in range(query2.shape[0]):
            # S subclass cognitive blocks applied to every input source.
            tempx1 = self.l2expert[i](x1_1)
            x1_2.append(tempx1)

            tempx2 = self.l2expert[i](x2_1)
            x2_2.append(tempx2)

            tempnan2 = self.l2expert[i](x_nan_2)
            x_nan_2_.append(tempnan2)

            # Subclass affinities (scaled dot products, Eq. 3 analogue).
            queryl2 = query2[i].unsqueeze(0).unsqueeze(0)

            attention1 = torch.sum(queryl2 * keys3_1, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionl2_1.append(attention1)

            attention2 = torch.sum(queryl2 * keys3_2, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionl2_2.append(attention2)

            attentionnan2 = torch.sum(queryl2 * keys_nan3, dim=-1) / torch.sqrt(torch.FloatTensor([768]).to(device))
            attentionlnan_2.append(attentionnan2)

        # Assemble the candidate outputs for the subclass routing:
        # [S subclass blocks] + [background block] (+ a zero block reserved
        # for the reconstruction router input layout).
        x1_2 = torch.stack(x1_2)
        x1_2_b = self.backexpert(x1_1).unsqueeze(0)
        xf = torch.zeros_like(x1_1).unsqueeze(0)
        x1_2_ = torch.cat((x1_2, x1_2_b), dim=0)
        xf_ = torch.cat((x1_2, xf, x1_2_b), dim=0)

        x2_2 = torch.stack(x2_2)
        x2_2_b = self.backexpert(x2_1).unsqueeze(0)
        x_f = torch.zeros_like(x2_1).unsqueeze(0)
        x_f_ = torch.cat((x2_2, x_f, x2_2_b), dim=0)

        x_nan_2_ = torch.stack(x_nan_2_)
        x_nan_2_b = self.backexpert(x_nan_2).unsqueeze(0)
        x_nan_f = torch.zeros_like(x_nan_2).unsqueeze(0)
        x_nan_f_ = torch.cat((x_nan_2_, x_nan_f, x_nan_2_b), dim=0)

        # Stacked subclass affinity maps: (S, seq_len, batch).
        attentionl2_1 = torch.stack(attentionl2_1)
        attentionl2_2 = torch.stack(attentionl2_2)
        attentionlnan_2 = torch.stack(attentionlnan_2)

        # ------------------------------------------------------------------
        # CDSE2 subclass routing (Eqs. 5-7): variance-gated affinities with
        # alpha2/theta2, modulated by the hierarchy coupling term
        # (1 + temperature * xadd_attention) injected from CDSE1 via c2m.
        # ------------------------------------------------------------------
        # routing for the basic data x
        v1_2 = torch.var(attentionl2_1, dim=0, unbiased=True).unsqueeze(0)
        v1_2_ = v1_2.repeat(attentionl2_1.shape[0], 1, 1)
        vecl2_1 = attentionl2_1 * torch.sigmoid(v1_2_ - self.theta2)  # (S, seq_len, batch)
        vecl2_2 = torch.ones_like(v1_2) * self.alpha2                 # (1, seq_len, batch)

        attx = torch.cat((vecl2_1, vecl2_2), dim=0)
        att1 = attx * (1 + self.tem * xadd_attention)  # HCKA-coupled weights
        attention_select_1 = torch.softmax(att1, dim=0)
        # Reconstruction router: known-class semantics are dynamically scaled
        # down before unknown-factor probing (Eqs. 8-10).
        attention_filt_1 = router_filt(attx, query2.shape[0], self.filtalpha)
        attention_select_1 = attention_select_1.unsqueeze(3)
        attention_filt_1 = attention_filt_1.unsqueeze(3)
        x1_2 = self.l2_2(x1_1 + torch.sum(x1_2_ * attention_select_1, dim=0))   # (seq_len, batch, 768)

        x1_2_f = self.l2_2(torch.sum(xf_ * attention_filt_1, dim=0))  # reconstructed semantics of x

        # routing + reconstruction for the LLM-UFDE data x'
        v2_2 = torch.var(attentionl2_2, dim=0, unbiased=True).unsqueeze(0)
        v2_2_ = v2_2.repeat(attentionl2_2.shape[0], 1, 1)
        vecl2_1_2 = attentionl2_2 * torch.sigmoid(v2_2_ - self.theta2)
        vecl2_2_2 = torch.ones_like(v2_2) * self.alpha2
        attx_ = torch.cat((vecl2_1_2, vecl2_2_2), dim=0)
        attention_filt_2 = router_filt(attx_, query2.shape[0], self.filtalpha)
        attention_filt_2 = attention_filt_2.unsqueeze(3)
        x2_2_f = self.l2_2(torch.sum(x_f_ * attention_filt_2, dim=0))  # reconstructed semantics of x'

        # routing + reconstruction for the background semantics
        vnan_2 = torch.var(attentionlnan_2, dim=0, unbiased=True).unsqueeze(0)
        vnan_2_ = v2_2.repeat(attentionlnan_2.shape[0], 1, 1)
        veclnan_1 = attentionlnan_2 * torch.sigmoid(vnan_2_ - self.theta2)
        veclnan_2 = torch.ones_like(vnan_2) * self.alpha2
        attnan_2 = torch.cat((veclnan_1, veclnan_2), dim=0)
        attention_filt_nan = router_filt(attnan_2, query2.shape[0], self.filtalpha)
        attention_filt_nan = attention_filt_nan.unsqueeze(3)
        x_nan_2_f = self.l2_2(torch.sum(x_nan_f_ * attention_filt_nan, dim=0))  # reconstructed background

        # ------------------------------------------------------------------
        # Known-class classification: encoded subclass-level semantics are
        # average-pooled and fed to the S binary classifiers (Eq. 5 head).
        # ------------------------------------------------------------------
        xcls = torch.mean(self.selectencoder(x1_2), dim=0)  # (batch, 768)

        # ------------------------------------------------------------------
        # MCLUFI (Section 3.5): the reconstruction router outputs are fused
        # by the fusion encoder; probes act on the filtered semantics.
        # xc1  -> filtered LLM-UFDE semantics (may contain unknown factors)
        # x_c1 -> filtered basic-data semantics (known classes only)
        # x_nan_c1 -> filtered background semantics
        # ------------------------------------------------------------------
        xc1 = self.filtencoder(x1_2_f)
        x_c1 = self.filtencoder(x2_2_f)
        x_nan_c1 = self.filtencoder(x_nan_2_f)
        extra_matrix = []
        x_c_e = []
        ood = []
        ood_ = []
        keys4_1 = self.keys1(x_c1)
        for i in range(self.gra):
            # Probe responses on the subclass definitions (Eq. 15 reference).
            tempextra = self.extraquery[i](query2).squeeze()  # (S,)
            extra_matrix.append(tempextra)

            # Probe activations on the filtered LLM-UFDE semantics.
            att_x = self.extraquery[i](self.keys1(xc1))  # (seq_len, batch, 1)
            ood.append(att_x.squeeze())

            # Probe activations on the filtered basic-data semantics.
            att_x_ = self.extraquery[i](keys4_1)  # (seq_len, batch, 1)
            tempx_c = torch.sum(x_c1 * att_x_, dim=0)  # (batch, 768)
            x_c_e.append(tempx_c)
            ood_.append(att_x_.squeeze())

        x_c_e = torch.stack(x_c_e)  # (P, batch, 768)
        ood = torch.stack(ood).permute(0, 2, 1)
        ood_ = torch.stack(ood_).permute(0, 2, 1)  # (P, seq_len, batch)

        extra_matrix = torch.stack(extra_matrix)  # (P, S)
        # Probe diversity loss (Eqs. 12-13): mean pairwise cosine similarity
        # between probe activation patterns; minimizing it decorrelates the
        # probes on both the LLM-UFDE data and the basic data.
        var_matrix2_1 = []
        var_matrix2_2 = []
        for i in range(extra_matrix.shape[0] - 1):
            for j in range(i + 1, extra_matrix.shape[0]):
                var_matrix2_1.append(F.cosine_similarity(ood[i], ood[j], dim=-1))
                var_matrix2_2.append(F.cosine_similarity(ood_[i], ood_[j], dim=-1))
        var_matrix2_1 = torch.stack(var_matrix2_1)
        var_matrix2_2 = torch.stack(var_matrix2_2)
        varloss = torch.mean(var_matrix2_2) + torch.mean(var_matrix2_1)

        # Instance-level pooled filtered semantics.
        xc_1 = torch.mean(xc1, dim=0)        # (batch, 768)
        x_c_1 = torch.mean(x_c1, dim=0)      # (batch, 768)
        x_nan_c1 = torch.mean(x_nan_c1, dim=0)  # (batch, 768)

        # Open-set detection statistic: LogSumExp-smoothed maximum probe
        # activation (differentiable Max approximation, Eqs. 16-19).
        maxattx = torch.logsumexp(ood, dim=2)      # (P, batch)
        maxattx_ = torch.logsumexp(ood_, dim=2)    # (P, batch)

        xlogit = torch.logsumexp(maxattx, dim=0)      # (batch,) on LLM-UFDE data
        x_logit = torch.logsumexp(maxattx_, dim=0)    # (batch,) on basic data

        # Reference statistic: max dot product between the subclass
        # definitions (query2) and the filtered known-class keys (Eq. 15).
        x_query2 = []
        for i in range(query2.shape[0]):
            tempquery2 = query2[i].unsqueeze(0).unsqueeze(0)
            att = torch.sum(tempquery2 * keys4_1, dim=-1).squeeze()
            att = torch.logsumexp(att, dim=0)
            x_query2.append(att)
        x_query2 = torch.stack(x_query2)
        x_query2 = torch.mean(x_query2, dim=0)  # (batch,)

        # Reconstruction component (Eq. 11): synthesized semantics of the
        # LLM-UFDE data, the basic data, and the background semantics.
        xfcompare_1 = self.active(xc_1)                          # (batch, 768)
        x_fcompare_1 = self.active(x_c_1)
        nan_value_1 = torch.mean(self.active(x_nan_c1), dim=0)   # (768,)

        l2out = []
        for clsl2 in range(self.semantic_sub.shape[0]):
            l2out.append(self.l2out[clsl2](xcls))
        l2out = torch.stack(l2out).permute(1, 0, 2).squeeze()  # (batch, S)

        # Reconstruction loss (Eq. 11): distance of potentially-unknown
        # (LLM-UFDE) semantics to the background should not be smaller than
        # the distance of known-only semantics to the background.
        dis_alpha = cossim2(xfcompare_1, nan_value_1, dim=-1).squeeze()
        dis_beta_1 = cossim2(x_fcompare_1, nan_value_1, dim=-1).squeeze()

        disloss_1 = torch.mean(F.relu(dis_alpha - dis_beta_1))
        # Novel-class exploration loss (Eqs. 14-15): the maximum probe
        # activation on the LLM-UFDE data must exceed the maximum subclass
        # definition affinity with the known-class data.
        disloss_2 = torch.mean(F.relu(x_query2 - x_logit))

        return l2out, [xlogit, disloss_1, disloss_2, varloss, dis_alpha]

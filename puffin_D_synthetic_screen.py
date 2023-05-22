import sys
import math
from typing import Union
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import pyBigWig
import tabix
import selene_sdk
from selene_sdk.targets import Target

sys.path.append("./utils/")
from selene_utils2 import *


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)


from torch import nn


class PuffinD(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(PuffinD, self).__init__()
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, 64, kernel_size=17, padding=8), nn.BatchNorm1d(64)
                ),
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
            ]
        )

        self.uplblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
            ]
        )

        self.upblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
            ]
        )

        self.downlblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=5),
                    nn.Conv1d(128, 128, kernel_size=17, padding=8),
                    nn.BatchNorm1d(128),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(128, 96, kernel_size=17, padding=8),
                    nn.BatchNorm1d(96),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=4),
                    nn.Conv1d(96, 64, kernel_size=17, padding=8),
                    nn.BatchNorm1d(64),
                ),
            ]
        )

        self.downblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(128, 128, fused=True), ConvBlock(128, 128, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(96, 96, fused=True), ConvBlock(96, 96, fused=True)
                ),
                nn.Sequential(
                    ConvBlock(64, 64, fused=True), ConvBlock(64, 64, fused=True)
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 10, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x):
        """Forward propagation of a batch."""
        out = x
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        return out

bignet = PuffinD()
bignet.load_state_dict(torch.load('./resources/puffin_D.pth'))
bignet.eval()
bignet.cuda()



class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(
        self,
        input_paths,
        features,
        shape,
        blacklists=None,
        blacklists_indices=None,
        replacement_indices=None,
        replacement_scaling_factors=None,
    ):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)]
        )
        self.shape = (len(input_paths), *shape)

    def get_feature_data(
        self, chrom, start, end, nan_as_zero=True, feature_indices=None
    ):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [
                    tabix.open(blacklist) for blacklist in self.blacklists
                ]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(
                        self.blacklists, self.blacklists_indices
                    ):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[
                                blacklist_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0) : int(e) - start] = 0
            else:
                for (
                    blacklist,
                    blacklist_indices,
                    replacement_indices,
                    replacement_scaling_factor,
                ) in zip(
                    self.blacklists,
                    self.blacklists_indices,
                    self.replacement_indices,
                    self.replacement_scaling_factors,
                ):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[
                            blacklist_indices,
                            np.fmax(int(s) - start, 0) : int(e) - start,
                        ] = (
                            wigmat[
                                replacement_indices,
                                np.fmax(int(s) - start, 0) : int(e) - start,
                            ]
                            * replacement_scaling_factor
                        )

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat



tfeature = GenomicSignalFeatures(
    [
        "../resources/agg.plus.bw.bedgraph.bw",
        "../resources/agg.encodecage.plus.v2.bedgraph.bw",
        "../resources/agg.encoderampage.plus.v2.bedgraph.bw",
        "../resources/agg.plus.grocap.bedgraph.sorted.merged.bw",
        "../resources/agg.plus.allprocap.bedgraph.sorted.merged.bw",
        "../resources/agg.minus.allprocap.bedgraph.sorted.merged.bw",
        "../resources/agg.minus.grocap.bedgraph.sorted.merged.bw",
        "../resources/agg.encoderampage.minus.v2.bedgraph.bw",
        "../resources/agg.encodecage.minus.v2.bedgraph.bw",
        "../resources/agg.minus.bw.bedgraph.bw",
    ],
    [
        "cage_plus",
        "encodecage_plus",
        "encoderampage_plus",
        "grocap_plus",
        "procap_plus",
        "procap_minus",
        "grocap_minus",
        "encoderampage_minus",
        "encodecage_minus",
        "cage_minus",
    ],
    (4000,),
    [
        "../resources/fantom.blacklist8.plus.bed.gz",
        "../resources/fantom.blacklist8.minus.bed.gz",
    ],
    [0, 9],
    [1, 8],
    [0.61357, 0.61357],
)


# select non-TSS locations

allcage = tfeature.get_feature_data("chr8", 0, 145138636)

t_signals_fwtss = []
t_signals_rctss = []
for t in np.arange(13808801, 30121240, 100):
    t_signals_fwtss.append(allcage[0, t - 150 : t + 150].max())
    t_signals_rctss.append(allcage[-1, t - 150 : t + 150].max())
t_list_filtered_rctssallowed = np.arange(13808801, 30121240, 100)[
    (np.array(t_signals_fwtss) < 0.01)
]
t_list_filtered = np.arange(13808801, 30121240, 100)[
    (np.array(t_signals_fwtss) < 0.01) * (np.array(t_signals_rctss) < 0.01)
]


np.random.seed(123)

genome = MemmapGenome(
    input_path="./resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    memmapfile="./resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
    blacklist_regions="hg38",
)



name_motifs = {
    "SP1": "GGGCGGGG",
    "SP1-rev": "CCCCGCCC",
    "CREB": "AATGACGTGA",
    "CREB-rev": "TCACGTCATT",
    "ETS": "ACTTCCGGT",
    "ETS-rev": "ACCGGAAGT",
    "5-splice-site": "AGGTAAG",
    "5-splice-site-rev": "CTTACCT",
    "NRF1": "GCGCATGCGC",
    "NRF1-rev": "GCGCATGCGC",
    "YY1": "AAAATGGCGGC",
    "YY1-rev": "GCCGCCATTTT",
    "TATA": "CTATAAAA",
    "TATA-rev": "TTTTATAG",
    "NFY": "CAGCCAATCAGA",
    "NFY-rev": "TCTGATTGGCTG",
    "ZNF143": "ACATTTCCCAGAATGC",
    "ZNF143-rev": "GCATTCTGGGAAATGT",
    "Empty": "",
}


unique_motifs = sorted(name_motifs.keys())
unique_motifs_len = [len(name_motifs[m]) for m in unique_motifs]

unique_motifs = sorted(name_motifs.keys())
unique_motifs_len = [len(name_motifs[m]) for m in unique_motifs]

# randomly choose 5 motifs
motif_choices = np.random.randint(0, len(unique_motifs), size=(10000000, 5))
# randomly choose motif positions
motif_positions = np.random.randint(-150, 140, size=(10000000, 5))


alts = []
alts_pos = []
alts_all = []

seqmuts = []
np.random.seed(0)
t_s = t_list_filtered[np.random.randint(0, len(t_list_filtered), 10000000)]
i_s = np.arange(10000000)

for t, i in zip(t_s, i_s):
    seq = genome.get_encoding_from_coords("chr8", t - 50000, t + 50000)

    seqmut = seq.copy()

    for j in range(motif_positions.shape[1]):
        seqmut[
            50000
            + motif_positions[i, j] : 50000
            + motif_positions[i, j]
            + unique_motifs_len[motif_choices[i, j]],
            :,
        ] = genome.sequence_to_encoding(name_motifs[unique_motifs[motif_choices[i, j]]])
    seqmuts.append(seqmut[None, :, :])
    if len(seqmuts) == 200:
        seqmuts = np.concatenate(seqmuts, axis=0)
        seqt = torch.FloatTensor(seqmuts).transpose(1, 2).cuda()
        with torch.no_grad():
            pred = bignet(seqt)
            alts.append(pred[:, :, 50000 - 100 : 50000 + 100].max(2)[0].cpu().numpy())
            alts_all.append(pred[:, :, 50000 - 100 : 50000 + 100].cpu().numpy())
            alts_pos.append(
                pred[:, :, 50000 - 100 : 50000 + 100].max(2)[1].cpu().numpy()
            )

        seqmuts = []

if len(seqmuts) > 0:
    seqmuts = np.concatenate(seqmuts, axis=0)
    seqt = torch.FloatTensor(seqmuts).transpose(1, 2).cuda()
    with torch.no_grad():
        pred = bignet(seqt)
        alts.append(pred[:, :, 50000 - 100 : 50000 + 100].max(2)[0].cpu().numpy())
        alts_all.append(pred[:, :, 50000 - 100 : 50000 + 100].cpu().numpy())
        alts_pos.append(pred[:, :, 50000 - 100 : 50000 + 100].max(2)[1].cpu().numpy())

alts = np.concatenate(alts)
alts_pos = np.concatenate(alts_pos)
alts_all = np.concatenate(alts_all)


torch.save(
    {
        "alts": alts,
        "alts_pos": alts_pos,
        "alts_all": alts_all,
        "t_s": t_s,
        "i_s": i_s,
        "motif_choices": motif_choices,
        "motif_positions": motif_positions,
        "unique_motifs": unique_motifs,
        "unique_motifs_len": unique_motifs_len,
    },
    "./data/puffin_D_denovotss_screen.pth",
    pickle_protocol=4,
)

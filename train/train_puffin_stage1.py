import os
import sys
import time
import pandas as pd
import numpy as np
import pyBigWig
import tabix
import torch
from torch_fftconv import FFTConv1d
from torch import nn
from selene_sdk.targets import Target
import selene_sdk

n_tsses = 40000
suffix = sys.argv[1]
modelstr = "stage1_" + str(n_tsses) + "_" + suffix
print(modelstr)
os.makedirs("./models", exist_ok=True)
sys.path.append("../utils/")
tsses = pd.read_table(
    "../resources/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v5.highconf.tsv",
    sep="\t",
)
genome = selene_sdk.sequences.Genome(
    input_path="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
)


class GenomicSignalFeatures(Target):
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


window_size = 4650
seqs = []
tars = []
for randi in range(n_tsses):
    chrm, pos, strand = (
        tsses["chr"].values[randi],
        tsses["TSS"].values[randi],
        tsses["strand"].values[randi],
    )
    offset = 1 if strand == "-" else 0
    seq = genome.get_encoding_from_coords(
        tsses["chr"][randi],
        tsses["TSS"][randi] - window_size // 2 + offset,
        tsses["TSS"][randi] + window_size // 2 + offset,
        tsses["strand"][randi],
    )
    tar = tfeature.get_feature_data(
        tsses["chr"][randi],
        tsses["TSS"][randi] - window_size // 2 + offset,
        tsses["TSS"][randi] + window_size // 2 + offset,
    )
    if strand == "-":
        tar = tar[::-1, ::-1]

    seqs.append(seq)
    tars.append(tar)
seqs = np.dstack(seqs)
tars = np.dstack(tars)
seqs = seqs.transpose([2, 1, 0])
tars = tars.transpose([2, 0, 1])
np.random.seed(1)
randinds = np.random.permutation(np.arange(n_tsses))
seqs = seqs[randinds, :]
tars = tars[randinds, :]
tsses_rand = tsses.iloc[randinds, :]
train_seqs = seqs[~tsses_rand["chr"].isin(["chr8", "chr9", "chr10"]).values, :]
valid_seqs = seqs[tsses_rand["chr"].isin(["chr10"]).values, :]
train_tars = tars[~tsses_rand["chr"].isin(["chr8", "chr9", "chr10"]).values, :]
valid_tars = tars[tsses_rand["chr"].isin(["chr10"]).values, :]




class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv = nn.Conv1d(4, 40, kernel_size=51, padding=25)

        self.conv_inr = nn.Conv1d(4, 10, kernel_size=15, padding=7)
        self.activation = nn.Sigmoid()

        self.deconv = FFTConv1d(80, 10, kernel_size=601, padding=300)

        self.deconv_inr = nn.ConvTranspose1d(20, 10, kernel_size=15, padding=7)
        self.softplus = nn.Softplus()

    def forward(self, x):
        y = torch.cat([self.conv(x), self.conv(x.flip([1, 2])).flip([2])], 1)
        y_inr = torch.cat(
            [self.conv_inr(x), self.conv_inr(x.flip([1, 2])).flip([2])], 1
        )

        yact = self.activation(y) * y
        y_inr_act = self.activation(y_inr) * y_inr

        y_pred = self.softplus(self.deconv(yact) + self.deconv_inr(y_inr_act))
        return y_pred


net = SimpleNet()

net.cuda()
net.train()
net.conv.weight.requires_grad = True
net.conv.bias.requires_grad = True
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.01)


def PseudoPoissonKL(lpred, ltarget):
    return ltarget * torch.log((ltarget + 1e-10) / (lpred + 1e-10)) + lpred - ltarget


def KL(pred, target):
    pred = (pred + 1e-10) / ((pred + 1e-10).sum(2)[:, :, None])
    target = (target + 1e-10) / ((target + 1e-10).sum(2)[:, :, None])
    return target * (torch.log(target + 1e-10) - torch.log(pred + 1e-10))


batchsize = 16

stime = time.time()
i = 0
bestcor = 0
past_losses = []
past_l2 = []
past_l1act = []
train_losses_stage1 = []
valid_losses_stage1 = []
weights = torch.ones(10).cuda()
bestloss = np.inf


def std2(x, axis, dim):
    return ((x - x.mean(axis=axis, keepdims=True)) ** dim).mean(axis=axis) ** (1 / dim)


while True:
    for j in np.random.permutation(range(train_seqs.shape[0] // batchsize)):
        sequence = train_seqs[j * batchsize : (j + 1) * batchsize, :, :]
        target = train_tars[j * batchsize : (j + 1) * batchsize, :, :]
        sequence = torch.FloatTensor(sequence)
        target = torch.FloatTensor(target)
        if torch.rand(1) < 0.5:
            sequence = sequence.flip([1, 2])
            target = target.flip([1, 2])
        optimizer.zero_grad()
        pred = net(torch.Tensor(sequence.float()).cuda())
        loss0 = (
            KL(pred[:, :, 325:-325], target.cuda()[:, :, 325:-325])
            * weights[None, :, None]
        ).mean()

        l2 = (
            (
                (
                    (net.deconv.weight[:, :, :-1] - net.deconv.weight[:, :, 1:]) ** 2
                ).mean(2)
                / (std2(net.deconv.weight, axis=2, dim=4) ** 2 + 1e-10)
            )
            * (weights)[:, None]
        ).mean()
        l1act = (net.deconv.weight.abs() * (weights)[:, None, None]).mean()
        l1motif = net.conv.weight.abs().mean()
        loss = (
            loss0
            + l2 * 2e-3
            + 1e-3
            * (
                PseudoPoissonKL(
                    pred[:, :, 325:-325] / np.log(10), target.cuda()[:, :, 325:-325]
                )
                * weights[None, :, None]
            ).mean()
            + l1act * 5e-5
            + l1motif * 4e-5
        )
        loss.backward()
        past_losses.append(loss0.detach().cpu().numpy())
        past_l2.append(l2.detach().cpu().numpy())
        past_l1act.append(l1act.item())
        optimizer.step()
        if i % 100 == 0:
            print("train loss:" + str(np.mean(past_losses[-100:])), flush=True)
            train_losses_stage1.append(np.mean(past_losses[-100:]))
            past_losses = []

        if i % 1000 == 0:
            with torch.no_grad():
                past_losses = []
                for j in range(valid_seqs.shape[0] // batchsize):
                    sequence = valid_seqs[j * batchsize : (j + 1) * batchsize, :, :]
                    target = valid_tars[j * batchsize : (j + 1) * batchsize, :, :]
                    sequence = torch.FloatTensor(sequence)
                    target = torch.FloatTensor(target)
                    if torch.rand(1) < 0.5:
                        sequence = sequence.flip([1, 2])
                        target = target.flip([1, 2])
                    optimizer.zero_grad()
                    pred = net(torch.Tensor(sequence.float()).cuda())
                    loss0 = (
                        KL(pred[:, :, 325:-325], target.cuda()[:, :, 325:-325])
                        * weights[None, :, None]
                    ).mean()
                    past_losses.append(loss0.detach().cpu().numpy())
            validloss = np.mean(past_losses)
            print("valid loss:" + str(validloss), flush=True)
            valid_losses_stage1.append(validloss)
            if validloss < bestloss:
                bestloss = validloss
                torch.save(net.state_dict(), "./models/" + modelstr + ".pth")
            if i // 1000 > 2000:
                sys.exit()

        i = i + 1

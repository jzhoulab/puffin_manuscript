import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import selene_sdk
from Bio.Seq import Seq


region_start = 22964801
region_end = 29963540
step_size = 2000
os.makedirs("./data/Insertion_screen/", exist_ok=True)

# start TSS in the TSS list
a = 0
# end TSS in the TSS list
b = 40000


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

    def forward(self, x, full_length_output=False):
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
        if full_length_output:
            return out
        else:
            return out


genome_path = "./resources/hg38.fa"
g = selene_sdk.sequences.Genome(input_path=genome_path)
model_path = "./resources/puffin_D.pth"

net = PuffinD()
net.load_state_dict(torch.load(model_path))
net.cuda()
net.eval()

# make WT prediciton
def predictWT(tsspos, chrm, strand, showRegion):
    """
    make prediciton for the target genomic location before the insertion
    """
    # define 100K region the model
    start100K = tsspos - 50000
    finish100K = start100K + 100000

    # get sequence
    seqWT1 = g.get_sequence_from_coords(chrm, start100K, finish100K)

    if strand == "-":
        seqWT1 = str(Seq(seqWT1).reverse_complement())
    seqWT = g.sequence_to_encoding(str(seqWT1))

    # calculate mean of predicted values for WT
    allpredWT = np.zeros((2, 100000))
    with torch.no_grad():
        pred = (
            net(torch.FloatTensor(seqWT[None, :, :]).transpose(1, 2).cuda())
            .cpu()
            .detach()
            .numpy()
        )
        pred_rc = (
            net(
                torch.FloatTensor(seqWT[None, ::-1, ::-1].copy()).transpose(1, 2).cuda()
            )
            .cpu()
            .detach()
            .numpy()
        )
        allpredWT[:, :] = 0.5 * pred[0, :, :] + 0.5 * pred_rc[0, :, :][::-1, ::-1]

    prediciton = allpredWT[0]

    return seqWT1, prediciton


def CreateInserMutations(sequence, motif):
    """
    create insertion mutant seaquence
    """
    s1 = sequence[int(len(motif) / 2) : 50000]
    s3 = sequence[50000 : -int(len(motif) / 2)]
    seq = s1 + motif + s3
    mutant = g.sequence_to_encoding(str(seq))
    return mutant


def MakePredicitonsBatch(mutant):
    """
    Make prediciotn for the batch of mutant sequences
    """
    # make predicition for mutant
    allpred = np.zeros((len(mutant), 10, 100000))
    with torch.no_grad():
        pred = (
            net(torch.FloatTensor(mutant[:, :, :]).transpose(1, 2).cuda())
            .cpu()
            .detach()
            .numpy()
        )
        pred_rc = (
            net(torch.FloatTensor(mutant[:, ::-1, ::-1].copy()).transpose(1, 2).cuda())
            .cpu()
            .detach()
            .numpy()
        )
        allpred[:, :, :] = 0.5 * pred[:, :, :] + 0.5 * pred_rc[:, :, :][:, ::-1, ::-1]
        # allpred = allpred[:,:5, :]

    return allpred[:, :, 50000 - 1000 : 50000 + 1000]


TSS_list_path = "./resources/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v5.tsv"
Tss10K = pd.read_csv(TSS_list_path, sep="\t")

region = 300
batch_size = 100


sources = range(len(Tss10K))[a:b]

for num, nTSS in zip(range(a, b), sources):

    print("sourceseq " + str(nTSS), flush=True)

    tsspos = Tss10K["TSS"].values[nTSS]
    chrm = Tss10K["chr"].values[nTSS]
    strand = Tss10K["strand"].values[nTSS]
    motif = g.get_sequence_from_coords(chrm, tsspos - region, tsspos + region)
    if strand == "-":
        motif = str(Seq(motif).reverse_complement())

    sample = 0
    mutantSeqBatch = np.zeros((batch_size, 100000, 4))
    line_length = int((region_end - region_start) / step_size) + 1

    line = np.zeros((line_length, 10, 6))
    line_exp = np.zeros((line_length, 10, 6))
    n = 0

    for p in range(region_start, region_end, step_size):

        WTsequence = g.get_sequence_from_coords("chr8", p - 50000, p + 50000)
        mutantSequence = CreateInserMutations(WTsequence, motif)

        mutantSeqBatch[sample, :] = mutantSequence
        sample += 1

        if sample % batch_size == 0 and sample != 0:

            pred = MakePredicitonsBatch(mutantSeqBatch)
            pred_exp = (10**pred) - 1

            for a in range(pred.shape[0]):
                for nb, b in enumerate([20, 50, 100, 200, 500, 1000]):

                    mf = np.mean(pred[a, :, 1000 - b : 1000 + b], axis=1)
                    mf_exp = np.mean(pred_exp[a, :, 1000 - b : 1000 + b], axis=1)

                    line[((batch_size * n) + a), :, nb] = mf
                    line_exp[((batch_size * n) + a), :, nb] = mf_exp

            mutantSeqBatch = np.zeros((batch_size, 100000, 4))
            sample = 0
            n += 1
    if sample != 0:
        mutantSeqBatch = mutantSeqBatch[:sample, :, :]
        pred = MakePredicitonsBatch(mutantSeqBatch)
        pred_exp = (10**pred) - 1

        for a in range(pred.shape[0]):
            for nb, b in enumerate([20, 50, 100, 200, 500, 1000]):

                mf = np.mean(pred[a, :, 1000 - b : 1000 + b], axis=1)
                mf_exp = np.mean(pred_exp[a, :, 1000 - b : 1000 + b], axis=1)

                line[((batch_size * n) + a), :, nb] = mf
                line_exp[((batch_size * n) + a), :, nb] = mf_exp

    np.save(output_path + "Ins_screen_TSS_" + str(nTSS), line)
    np.save(output_path + "Ins_screen_exp_TSS_" + str(nTSS), line_exp)

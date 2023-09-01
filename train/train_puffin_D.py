import time
import numpy as np
import tabix
import torch
import selene_sdk
import pyBigWig
from torch import nn
from scipy.special import softmax
from matplotlib import pyplot as plt
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader
torch.set_default_tensor_type('torch.FloatTensor')


seed=3
modelstr = 'puffin_D'

class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
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
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True
            
        wigmat = np.vstack([c.values(chrom, start, end, numpy=True)
                           for c in self.data])
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat


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
        self.uplblocks = nn.ModuleList([
                nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),

            nn.Sequential(
            nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True)),

            nn.Sequential(
            ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks = nn.ModuleList([
    
            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
            nn.Conv1d(128, 96, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(96, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),


        ])

        self.downblocks = nn.ModuleList([
                nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True))


        ])

        self.uplblocks2 = nn.ModuleList([
    
            nn.Sequential(
            nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

        ])

        self.upblocks2 = nn.ModuleList([
    
            nn.Sequential(
            ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks2 = nn.ModuleList([
    
            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
            nn.Conv1d(128, 96, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(96, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),


        ])

        self.downblocks2 = nn.ModuleList([
                nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True))


        ])
        self.final =  nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 10, kernel_size=1),
            nn.Softplus())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = x
        encodings = []
        for i, lconv, conv in zip(np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        return out
 




tfeature = GenomicSignalFeatures(["./resources/agg.plus.bw.bedgraph.bw",
"./resources/agg.encodecage.plus.v2.bedgraph.bw",
"./resources/agg.encoderampage.plus.v2.bedgraph.bw",
"./resources/agg.plus.grocap.bedgraph.sorted.merged.bw",
"./resources/agg.plus.allprocap.bedgraph.sorted.merged.bw",
"./resources/agg.minus.allprocap.bedgraph.sorted.merged.bw",
"./resources/agg.minus.grocap.bedgraph.sorted.merged.bw",
"./resources/agg.encoderampage.minus.v2.bedgraph.bw",
"./resources/agg.encodecage.minus.v2.bedgraph.bw",
"./resources/agg.minus.bw.bedgraph.bw"],
                               ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
,'encoderampage_minus', 'encodecage_minus',
'cage_minus'],
                               (100000,),
                               ["./resources/fantom.blacklist8.plus.bed.gz","./resources/fantom.blacklist8.minus.bed.gz"],
                               [0,9], [1,8], [0.61357, 0.61357])


weights = torch.ones(10).cuda()


genome = selene_sdk.sequences.Genome(
                    input_path='./resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
                    blacklist_regions= 'hg38'
                )

noblacklist_genome = selene_sdk.sequences.Genome(
                    input_path='./resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa' )


sampler = RandomPositionsSampler(
                reference_sequence= genome,
                target= tfeature,
                features = [''],
                test_holdout=['chr8', 'chr9'],
                validation_holdout= ['chr10'],
                sequence_length= 100000,
                center_bin_to_predict= 100000,
                position_resolution=1,
                random_shift=0,
                random_strand=False
)



sampler.mode="train"
dataloader = SamplerDataLoader(sampler, num_workers=32, batch_size=32, seed=seed)


def figshow(x,np=False):
    if np:
        plt.imshow(x.squeeze())
    else:
        plt.imshow(x.squeeze().cpu().detach().numpy())
    plt.show()



try:
    net = torch.load('./models/'+modelstr+'.checkpoint')
except:
    print("pretrained model not found")
    net = nn.DataParallel(PuffinD())


net.cuda()
net.train()

params = [p for p in net.parameters() if p.requires_grad]

from torch.optim.lr_scheduler import ReduceLROnPlateau
optimizer = torch.optim.Adam(params,lr=0.005)
try:
     temp = torch.load('./models/'+modelstr+'.optimizer')
     optimizer.load_state_dict(temp.state_dict())
except:
    print("pretrained optimizer not found")
    
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=10, threshold=0)

from scipy.stats import spearmanr


def PseudoPoissonKL(lpred, ltarget):
    return (ltarget * torch.log((ltarget+1e-10)/(lpred+1e-10)) + lpred - ltarget)


i=0
past_losses=[]
firstvalid=True
bestcor=0
while True:
    for sequence, target in dataloader:
            if torch.rand(1)<0.5:
                sequence = sequence.flip([1,2])
                target = target.flip([1,2])

            optimizer.zero_grad()
            pred = net(torch.Tensor(sequence.float()).transpose(1,2).cuda())
            loss = (PseudoPoissonKL(pred, target.cuda()) * weights[None,:,None]).mean()
            loss.backward()
            past_losses.append(loss.detach().cpu().numpy())

            optimizer.step()

            del pred
            del loss
            
            if i % 500 ==0:
                print("train loss:"+str(np.mean(past_losses[-500:])),flush=True)


            if i % 500 == 0:
                torch.save(net, './models/'+modelstr+'.checkpoint')
                torch.save(optimizer, './models/'+modelstr+'.optimizer')

            rstate_saved = np.random.get_state()
            if i % 8000 == 0:
                if firstvalid:
                    validseq = noblacklist_genome.get_encoding_from_coords("chr10", 0, 114364328)
                    validcage = tfeature.get_feature_data("chr10", 0, 114364328)
                    firstvalid = False
                net.eval()
                print(validseq.shape, flush=True)
                with torch.no_grad():
                    validpred = np.zeros((10, 114364328))
                    kllosses = []
                    for ii in np.arange(0, 114364328, 50000)[:-2]:
                        pred = (
                            net(
                                torch.FloatTensor(validseq[ii : ii + 100000, :][None, :, :])
                                .transpose(1, 2)
                                .cuda()
                            )
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        pred2 = (
                            net(
                                torch.FloatTensor(validseq[ii : ii + 100000, :][None, ::-1, ::-1].copy())
                                .transpose(1, 2)
                                .cuda()
                            )
                            .cpu()
                            .detach()
                            .numpy()[:, ::-1, ::-1]
                        )

                        validpred[:, ii + 25000 : ii + 75000] = (
                            pred[0, :, 25000:75000] * 0.5 + pred2[0, :, 25000:75000] * 0.5
                        )



                validcor = (
                    np.corrcoef(validpred[0, :ii], validcage[0, :ii])[0, 1] * 0.5
                    + np.corrcoef(validpred[-1, :ii], validcage[-1, :ii])[0, 1] * 0.5
                )
                validcor2 = (
                    np.corrcoef(validpred[1, :ii], validcage[1, :ii])[0, 1] * 0.5
                    + np.corrcoef(validpred[-2, :ii], validcage[-2, :ii])[0, 1] * 0.5
                )
                validcor3 = (
                    np.corrcoef(validpred[2, :ii], validcage[2, :ii])[0, 1] * 0.5
                    + np.corrcoef(validpred[-3, :ii], validcage[-3, :ii])[0, 1] * 0.5
                )
                validcor4 = (
                    np.corrcoef(validpred[3, :ii], validcage[3, :ii])[0, 1] * 0.5
                    + np.corrcoef(validpred[-4, :ii], validcage[-4, :ii])[0, 1] * 0.5
                )
                validcor5 = (
                    np.corrcoef(validpred[4, :ii], validcage[4, :ii])[0, 1] * 0.5
                    + np.corrcoef(validpred[-5, :ii], validcage[-5, :ii])[0, 1] * 0.5
                )
                print("Cor {0} {1} {2} {3} {4}".format(validcor, validcor2, validcor3, validcor4, validcor5))

                net.train()
                if bestcor < validcor + validcor2 + validcor3 + validcor4 + validcor5:
                    bestcor = validcor + validcor2 + validcor3 + validcor4 + validcor5
                    torch.save(net, './models/'+modelstr+'.best.checkpoint')
                    torch.save(optimizer, './models/'+modelstr+'.best.optimizer')
            i+=1


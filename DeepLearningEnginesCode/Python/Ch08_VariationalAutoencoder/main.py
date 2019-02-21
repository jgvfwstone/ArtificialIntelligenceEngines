from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from torch.autograd import Variable

# connections through the autoencoder bottleneck
# in the pytorch VAE example, this is 20
ZDIMS = 20
BATCH_SIZE = 128

numepochs=1

parser = argparse.ArgumentParser(description='VAE MNIST Example')

parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=numepochs, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(784, 400)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(400, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections
        
       # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(ZDIMS, 400)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x):
        
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
##        
##        """THE REPARAMETERIZATION IDEA:
##
##        For each training sample (we get 128 batched at a time)
##
##        - take the current learned mu, stddev for each of the ZDIMS
##          dimensions and draw a random sample from that distribution
##        - the whole network is trained so that these randomly drawn
##          samples decode to output that looks like the input
##        - which will mean that the std, mu will be learned
##          *distributions* that correctly encode the inputs
##        - due to the additional KLD term (see loss_function() below)
##          the distribution will tend to unit Gaussians
##
##        Parameters
##        ----------
##        mu : [128, ZDIMS] mean matrix
##        logvar : [128, ZDIMS] variance matrix
##
##        Returns
##        -------
##
##        During training random sample from the learned ZDIMS-dimensional
##        normal distribution; during inference its mean.
##
##        """
        if self.training:
            std = torch.exp(0.5*logvar)
             # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = torch.randn_like(std)
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)
        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalise by same number of elements as in reconstruction: JVS this line is missing from pytorch repo.
    # KLD /= BATCH_SIZE * 784 # JVS thinks botha introduced this line.
    KLD /= BATCH_SIZE # JVS maths says this is correct, but worrying that it was not in Kingma's code.
    # JVS Kingma's repo does not have above line. https://github.com/dpkingma/examples/blob/master/vae/main.py
    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    
    return BCE + KLD

# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    fig = plt.figure(9)
    plt.clf()
    ax=fig.add_subplot(111)
    ax.set_xlabel('Batch number')
    ax.set_ylabel('ELBO')
    plt.ylim(0,1e7)
    plt.xlim(0,len(train_loader))
    
    train_losses=[]
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # data = data.to(device)
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        train_losses.append(train_loss)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            plt.ion()
            ax.plot(train_losses,c='black')
            plt.draw()
            plt.pause(0.001)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                # for the first 128 batch of the epoch, show the first 8 input digits
                # with right below them the reconstructed output digits
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        test(epoch)
        train(epoch)
        with torch.no_grad():
            # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
            # digits in latent space
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            # save out as an 8x8 matrix of MNIST digits
            # this will give you a visual idea of how well latent space can generate things
            # that look like digits
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
            
#            fig = plt.figure(1)
#            plt.clf()
#            ax = fig.add_subplot(111)
#            a = model.fc1.weight.detach()
#            for i in range(0,5):
#                #plt.imshow(model.fc1.weight.detach())
#                b=a[i]
#                c=b.view(28,28)
#                plt.imshow(c,cmap="gray")
#                plt.show()
                
            # JVS plot feature of unit in 1st hidden layer
            fig, axes = plt.subplots(4, 4)
            fig.subplots_adjust(left=None, bottom=None, 
                                right=None, top=None, 
                                wspace=0, hspace=0)
            a = model.fc1.weight.detach()
            count=0
            for x in range(0,4):
                for y in range(0,4):
                    count=count+1
                    b=a[count]
                    c=b.view(28,28)
                    ax = axes[x,y]
                    ax.imshow(c,cmap="gray")
                    ax.set_xticks(())
                    ax.set_yticks(())


#    for i in range(0, 5):  plt.imshow(lum_img, cmap="hot")
#        b=a[i]
#        b.shape
#        c=b.view(28,28)
#        plt.imshow(c)
#        plt.show()
#        print('Number =',i)

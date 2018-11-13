import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TopicVAE(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        ac = net_arch
        self.net_arch = net_arch
        # encoder
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)             # 1995 -> 100
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(ac.num_topic, ac.num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)                      # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var    = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        # initialize decoder weight
        if ac.init_mult != 0:
            self.decoder.weight.data.uniform_(0, ac.init_mult)
            
    def encoder(self, input):
        assert input.shape[1] == self.net_arch.num_input, "input isn't batch size x vocab size"
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        return posterior_mean, posterior_logvar, posterior_var
    
    def reparameterize(self, input, posterior_mean, posterior_var):
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        return z
      
    def generative(self, z):
        raise NotImplementedError
        
    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        posterior_mean, posterior_logvar, posterior_var = self.encoder(input)
        z = self.reparameterize(input, posterior_mean, posterior_var)
        recon = self.generative(z)
        assert recon.shape[1] == self.net_arch.num_input, "output isn't batch size x vocab size"
        
        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1) # vector with batch-size number of elements
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean) # batch-size x num_topics
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic )

        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean() # averaged over all the documents in the batch (1/batch_size)*sum
        else:
            return loss
          
    
          
def train(model, args, optimizer, dataset):
    '''
    model - object of class TopicVAE
    args - dict of args
    optimizer - nn.optim
    dataset - docs x vocab tensor document term matrix
    '''
    for epoch in range(args.num_epoch):
        all_indices = torch.randperm(dataset.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            if not args.nogpu: batch_indices = batch_indices.cuda()
            input = Variable(dataset[batch_indices])
            recon, loss = model(input, compute_loss=True)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))

    return model
            
            
class ProdLDA(TopicVAE):
    def __init__(self, net_arch):
        super().__init__(net_arch)
        
    def generative(self, z):
        assert z.shape[1] == self.net_arch.num_topic, "hidden variable z (from TR) isn't batch size x num_topic"    
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        assert p.shape[1] == self.net_arch.num_topic, "p (theta) isn't same size as z"
        recon = F.softmax(self.decoder_bn(self.decoder(p)))             # reconstructed distribution over vocabulary
        return recon

    def get_beta(self):
        return self.decoder.weight.data.cpu().numpy()
    

class LDA(TopicVAE):
    def __init__(self, net_arch):
        super().__init__(net_arch)
        self.beta = nn.Parameter(torch.randn([self.net_arch.num_input, self.net_arch.num_topic]))
        self.beta_bn = nn.BatchNorm1d(self.net_arch.num_topic)
        
    def generative(self, z):
        assert z.shape[1] == self.net_arch.num_topic, "hidden variable z (from TR) isn't batch size x num_topic"    
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        assert p.shape[1] == self.net_arch.num_topic, "p (theta) isn't same size as z"
        recon = F.softmax(self.beta_bn(self.beta), dim=0).mm(p.t()).t()
        return recon

    def get_beta(self):
        return self.decoder.weight.data.cpu().numpy()


class GSMLDA(TopicVAE):
    def __init__(self, net_arch, pretrained_embed_matrix=None):
        super().__init__(net_arch)
        if pretrained_embed_matrix is None:
            self.word_embedding = nn.Embedding(net_arch.num_input, net_arch.embedding_dim)
        else:
            assert net_arch.embedding_dim == pretrained_embed_matrix.shape[1], "embedding dimension doesn't match embedding matrix"
            self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embed_matrix, dtype = torch.float32), 
                freeze = net_arch.freeze)
            # later on, option to change freeze=False
        self.word_embedding_bn = nn.BatchNorm1d(net_arch.embedding_dim)
        self.topic_embedding = nn.Embedding(self.net_arch.num_topic, net_arch.embedding_dim)
        self.topic_embedding_bn = nn.BatchNorm1d(net_arch.embedding_dim)
        self.beta = torch.zeros([self.net_arch.num_topic, self.net_arch.num_input], dtype = torch.float32)

    def generative(self, z):
        assert z.shape[1] == self.net_arch.num_topic, "hidden variable z (from TR) isn't batch size x num_topic"    
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        assert p.shape[1] == self.net_arch.num_topic, "p (theta) isn't same size as z"
        # do reconstruction
        word_vec = self.word_embedding_bn(self.word_embedding.weight)
        topic_vec = self.topic_embedding_bn(self.topic_embedding.weight)
        self.beta = F.softmax(word_vec.mm(topic_vec.t()), dim = 0) # Vx100 times 100xK => beta is VxK
        recon = p.mm(self.beta.t())         # reconstructed distribution over vocabulary
        # p is batchxK so batchxK times KxV => batchxV
        return recon

    def get_beta(self):
        return self.beta.detach().numpy()
      

class GSMProdLDA(TopicVAE):
    def __init__(self, net_arch, pretrained_embed_matrix=None):
        super().__init__(net_arch)
        if pretrained_embed_matrix is None:
            self.word_embedding = nn.Embedding(net_arch.num_input, net_arch.embedding_dim)
        else:
            assert net_arch.embedding_dim == pretrained_embed_matrix.shape[1], "embedding dimension doesn't match embedding matrix"
            self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embed_matrix, dtype = torch.float32), 
                freeze = net_arch.freeze)
            # later on, option to change freeze=False
        self.word_embedding_bn = nn.BatchNorm1d(net_arch.embedding_dim)
        self.topic_embedding = nn.Embedding(self.net_arch.num_topic, net_arch.embedding_dim)
        self.topic_embedding_bn = nn.BatchNorm1d(net_arch.embedding_dim)
        self.beta = torch.zeros([self.net_arch.num_topic, self.net_arch.num_input], dtype = torch.float32)

    def generative(self, z):
        assert z.shape[1] == self.net_arch.num_topic, "hidden variable z (from TR) isn't batch size x num_topic"    
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        assert p.shape[1] == self.net_arch.num_topic, "p (theta) isn't same size as z"
        # do reconstruction
        word_vec = self.word_embedding_bn(self.word_embedding.weight)
        topic_vec = self.topic_embedding_bn(self.topic_embedding.weight)
        self.beta = word_vec.mm(topic_vec.t()) # Vx100 times 100xK => beta is VxK
        recon = F.softmax(p.mm(self.beta.t()), dim = 0)         # reconstructed distribution over vocabulary
        # p is batchxK so batchxK times KxV => batchxV
        return recon

    def get_beta(self):
        return self.beta.detach().numpy()
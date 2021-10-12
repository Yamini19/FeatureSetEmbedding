import torch.nn as nn
import torch.nn.functional as F
import torch

class FeatureSetEmbedder(nn.Module):
    def __init__(self, max_n_features, embedding_dim, n_hidden=128, n_output=2):
        super(FeatureSetEmbedder, self).__init__()
        self.max_n_features = max_n_features
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(max_n_features, embedding_dim)
        self.pooling = nn.AvgPool2d((1,embedding_dim))
        self.linear1 = nn.Linear(max_n_features, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)
        
        
    def forward(self, inputs):
        # inputs is one instance with n_features x 2
        # first column is the feature indexes
        #  -> a zero in this column means that the feature is not present
        # second column is the feature values
        indexes = inputs[0][:, 0].type(torch.LongTensor)
        #values = inputs[:, 1]
        embeds = self.embeddings(indexes).squeeze() # is now n_features x embedding_dim
        x =None
        for inp in inputs:
            values = inp[:, 1]
            if x is None:
                x = (embeds * values.view(self.max_n_features, 1)).reshape(1,4,3)
            else: 
                temp = (embeds * values.view(self.max_n_features, 1)).reshape(1,4,3)
                x = torch.cat((x,temp),0)
        
        x= self.pooling(x)
        x =x.reshape(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x,dim=1)
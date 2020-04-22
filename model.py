import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # create embeddings regarding size of vocab and given embed size
        self.embed = nn.Embedding(vocab_size, embed_size) 
        #Generate the LSTM cell according to the arguments i.e embed_size, hidden layers etc.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        #Makeing hidden layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        #initialise hidden matrix of zeros
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
        
    
    def forward(self, features, captions):
        
        #make caption embeddings
        cap = self.embed(captions[:, :-1])
        embedding = torch.cat((features.unsqueeze(1), cap), 1)
        
        #input the embeddings through the LSTM cell.   
        lstm_output, self.hidden = self.lstm(embedding)
        output = self.linear(lstm_output)
        #return the output of the Feed forward 
        return output

    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        prediction = []#list for the predictions made by the decoder
        
        for i in range(max_len):
            lstm_output, hidden = self.lstm(inputs, hidden)#run through lstm
            output = self.linear(lstm_output.squeeze(1))#make a linear vector 
            target_idx = output.max(1)[1]#select the max out of predictions
            prediction.append(target_idx.item())#append that specific index of max prediction
            inputs = self.embed(target_idx). unsqueeze(1)#for new inputs
        #return list of predictions
        return prediction
        
        
        
        
        
        
        
        
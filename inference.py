import pandas as pd
import numpy as np
import torch
from src.models import CNN
from src.models import LSTMModel
from src.models import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM().to(device)
model.load_state_dict(torch.load('./best.pth')['model'])

test = pd.read_csv('./data/test.csv')

if __name__ == '__main__':
	p_LVD = []
	for uid, bef in zip(test['UID'].to_list(), test['BEF'].to_list()):
		x = pd.read_csv(f'./data/ecg/{uid}.csv').to_numpy()
		x = torch.FloatTensor(x).to(device).unsqueeze(0).unsqueeze(0)
		bef = torch.FloatTensor([bef]).to(device)
		with torch.no_grad():
			model.eval() 
			output = model(x, bef)
		output = output[0][0].to('cpu')
		p_LVD.append(output.item())
	
	answer = pd.read_csv('../sample_submission.csv')
	
	answer['p_LVD'] = p_LVD
	# answer['p_LVD'] = answer['p_LVD'].apply(lambda x: 90 if 90<x else x)
	# answer['p_LVD'] = answer['p_LVD'].apply(lambda x: 10 if 10>x else x)
	
	answer['p_LVD'] = answer['p_LVD'].apply(lambda x: 1-(((x-10)*1.25)/100))
	# answer['p_LVD'] = answer['p_LVD'].apply(lambda x: 1 if x <= 35 else 0.5-((x-35)/110))
  
  answer.to_csv('answer.csv', index=False)

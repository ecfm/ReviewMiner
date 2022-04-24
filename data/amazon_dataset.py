import torch.utils.data
# import pandas as pd

import data.util
# def read_csv(filename, chunksize=1000000):
#   reader = pd.read_csv(filename, chunksize=chunksize)
#   df = pd.DataFrame()
#   for chunk in reader:
#     df = pd.concat([df, chunk])
#   return df

class AmazonDataset(torch.utils.data.Dataset):
  def __init__(self, filename, preprocess=lambda x: x, sort=False):
    df = data.util.read_csv("/Users/STSadmin/Downloads/sample.csv")
    self.users = df['reviewerID']
    for u in self.users:
      if u not in data.util.uid_dict:
        data.util.uid_dict[u] = len(data.util.uid_list)
        data.util.uid_list.append(u)
    self.products = df['asin']
    for p in self.products:
      if p not in data.util.pid_dict:
        data.util.pid_dict[p] = len(data.util.pid_list)
        data.util.pid_list.append(p)
    self.reviews = df['reviewText']
    self.preprocess = preprocess

  def __getitem__(self, idx):
    review = self.reviews[idx]
    return {
        'u_id': data.util.uid_dict[self.users[idx]], 
        'p_id': data.util.pid_dict[self.products[idx]], 
        'review': self.preprocess(review)
    }
  def __len__(self):
    return len(self.reviews)


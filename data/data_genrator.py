import numpy as np
import pandas as pd
import os



def generate_data_if_not_exists(path, n_samples=100000):
    if not os.path.exists(path):
        # generated data randomly
        data = (np.random.rand(n_samples, 8) * np.array([100000, 1000000, 10000, 10000, 10000, 1000000, 1.4, 1.1])).astype(int).astype(str)
        csv  = pd.DataFrame(data, 
                    columns=['unique_id', 'item_id', 'shop_id', 'cate_id', 'brand_id', 'qid', 'click', 'pay'], dtype=str)
        
        csv.to_csv(path, index=False, header=None)




if __name__ == '__main__':
    generate_data_if_not_exists('./data/train.csv', n_samples=100000)
    generate_data_if_not_exists('./data/test.csv',  n_samples=10000)


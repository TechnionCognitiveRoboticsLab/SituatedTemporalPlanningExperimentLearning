import pandas
import numpy
import torch
from torch.utils.data import Dataset, ConcatDataset, RandomSampler, DataLoader, WeightedRandomSampler
import random
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

#from speechbrain.dataio.sampler import ConcatDatasetBatchSampler

# Updated with SOLVED/UNSOLVED to support learning from unsolved instances

class SingleFileSearchDumpDataset(Dataset):    
    def __init__(self, datafilename : str, solved : bool, 
                 height : int = 3, 
                 seq_len : int = 10,
                 labels : Optional[List[int]] = None):
        self.datafilename = datafilename
        self.solved = solved
        self.height = height
        self.seq_len = seq_len
        self.labels = labels
        
        self.basic_header_names = []
        self.df = pandas.read_csv(self.datafilename, sep='\t', compression="infer", index_col="id")
               
        for c in self.df.columns:
            if c != 'id' and c != 'path':                
                self.basic_header_names.append(c)

    def generate_row_X(self, row):
        path = row.path
        crow = row[self.basic_header_names].to_numpy(dtype=numpy.float64)
        megarow_list = [#[data_row.search_algorithm, data_row.heuristic, data_row.domain, data_row.problem, data_row.search_dump_file],
                        crow]
        if isinstance(path, str):                    
            nodes = list(filter(lambda node: node != "", path.split(",")[::-1]))[:self.height]                    
        else:
            nodes = []
        for node in nodes:                    
            node_row = self.df.loc[node][self.basic_header_names].to_numpy(dtype=numpy.float64)
            if node_row.ndim > 1:
                node_row = node_row[0,:]
            megarow_list.append(node_row)                        
        megarow_list.append([0.0] * len(self.basic_header_names) * (self.height - len(nodes)))
        megarow = torch.as_tensor(numpy.concatenate(megarow_list,axis=None)).float()
        return megarow

    def label(self, row, idx):
        #TODO: handle multiple labels
        if self.labels is None:
            label = torch.as_tensor( row.N / (len(self.df.index) - 1) ).float()
        else:
            label = torch.as_tensor([self.solved and idx + horizon > len(self.df) for horizon in self.labels], dtype=torch.bool)            
        return label

    def __getitem__(self, idx):        
        row = self.df.iloc[idx]
        label = self.label(row, idx)
        Xs = []  
        current_index = idx      
        while current_index >= 0 and current_index > idx - self.seq_len:
            row = self.df.iloc[current_index]
            Xs.append(self.generate_row_X(row))
            current_index = current_index - 1
        while current_index > idx - self.seq_len:
            Xs.append(torch.as_tensor([0.0] * len(self.basic_header_names) * (self.height + 1)))
            current_index = current_index - 1
        return torch.stack(Xs), label
    
    def __len__(self):        
        return len(self.df)

class SearchDumpDataset(ConcatDataset):    
    """Class for a dataet from search dump files.
    Params:
    height - how high up to go from each node up to the root
    seq_len - how far back in search history to go back
    labels - if None, use normalized search progress as labels, train only on solved instances.
             if given list of ints, train multiple binary concepts on whether the problem will be solved in X expansions, for every X in the list
    other parameters are filters for the dataset         
    """
    def __init__(self, datafilename : str, 
                 height : int = 3, 
                 seq_len : int = 10,
                 labels : Optional[List[int]] = None, 
                 min_expansions : int = 1000, 
                 max_expansions : int = 1000000, 
                 search_algorithm : str = "",
                 heuristic : str = "",
                 domain : str = "",
                 not_domain : bool = False):
        self.datafilename = datafilename
        self.height = height
        self.seq_len = seq_len
        self.labels = labels
        self.min_expansions = min_expansions
        self.max_expansions = max_expansions
        self.search_algorithm = search_algorithm
        self.heuristic = heuristic
        self.domain = domain
        self.not_domain = not_domain

        df = pandas.read_csv(datafilename, skipinitialspace = True)    
        df = df[(df.expansions <= self.max_expansions) & (df.expansions >= self.min_expansions)]
        if self.domain != "":
            if self.not_domain:
                df = df[df.domain != self.domain]
            else:
                df = df[df.domain == self.domain]
        if self.search_algorithm != "":
            df = df[df.search_algorithm == self.search_algorithm]
        if self.heuristic != "":
            df = df[df.heuristic == self.heuristic]
        if self.labels is None:
            df = df[df.solved == 1]
        self.data_df = df

        datasets = []
        for _, row in self.data_df.iterrows():
            datasets.append(SingleFileSearchDumpDataset(row.search_dump_file, row.solved, height, seq_len, labels))

        ConcatDataset.__init__(self, datasets)

    def get_domains(self):
        return numpy.unique(self.data_df.domain.values)
        

class SearchDumpDatasetSampler(WeightedRandomSampler):    
    def __init__(self, ds : SearchDumpDataset, num_samples: Optional[int] = None, generator=None):
        self.ds = ds
        self._num_samples = num_samples
        self.generator = generator
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:            
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))        

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.ds)
        return self._num_samples
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.ds)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        
        for _ in range(self.num_samples):
            dataset_index = torch.randint(high=len(self.ds.datasets), size=(1,), generator=generator).tolist()[0]
            row_index = torch.randint(high=len(self.ds.datasets[dataset_index]), size=(1,), generator=generator).tolist()[0]            
            yield self.ds.cumulative_sizes[dataset_index] - (row_index + 1)
        
        



def main():
    random.seed(42)
    filename="/home/karpase/git/SituatedTemporalPlanningExperiment/situated_dataset.csv"
    
    #filename=sys.argv[1]
    ds = SearchDumpDataset(filename, height=3, seq_len = 10, labels=[1000, 2000, 4000, 8000])

    print(ds.get_domains())

    
    print(len(ds))
    print(ds[0])
    print(ds[1])
    print(ds[3])

    sampler = SearchDumpDatasetSampler(ds, num_samples=100)
    dataloader = DataLoader(ds, sampler=sampler)
    
    for batch in sampler:
        print(batch)

    for X, y in dataloader:
        print(X,y)

    

if __name__ == "__main__":
    # $1 - the data.csv file generates from the experiment
    main()
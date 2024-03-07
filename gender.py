import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv('/shared/data/ChestX-ray14/Data_Entry_2017.csv')

positive = df[(df['Finding Labels'].str.contains("Atelectasis")) & (df['View Position']=="PA")]
positive = positive[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Gender']].rename(columns={"Image Index": "path", "Finding Labels": "class", "Patient ID": "pid"})
positive['class'] = 'Atelectasis'
negative = df[(df['Finding Labels'].str.contains("No Finding")) & (df['View Position']=="PA")]
negative = negative[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Gender']].rename(columns={"Image Index": "path", "Finding Labels": "class", "Patient ID": "pid"})

positive_M = positive[positive['Patient Gender']=='M']
positive_M = positive_M.sample(n=400)
positive_F = positive[positive['Patient Gender']=='F'] 
positive_F = positive_F.sample(n=200)
negative_M = negative[negative['Patient Gender']=='M']
negative_M = negative_M.sample(n=200)
negative_F = negative[negative['Patient Gender']=='F'] 
negative_F = negative_F.sample(n=200)

gss = GroupShuffleSplit(n_splits=2, test_size = 0.5, random_state=42)
train_index, test_index = next(gss.split(positive_M['path'], positive_M['class'], groups=positive_M['pid']))
positive_M_train = positive_M.iloc[train_index].reset_index(drop=True)
positive_M_test = positive_M.iloc[test_index].reset_index(drop=True)

test_set = pd.concat([positive_M_test, negative_F], ignore_index=True, sort=False)
test_set.to_csv('/home/doju/data/chest14/gender/test.csv')

p = 0
gkf = GroupShuffleSplit(n_splits=5, test_size = 0.15, random_state=42)
train_data = pd.concat([positive_F.sample(frac=p), positive_M_train.sample(frac=1-p), negative_M], ignore_index=True, sort=False)
print(train_data)
fold_no = 1  # initialize fold counter
for train_index, val_index in gkf.split(train_data['path'], train_data['class'], groups=train_data['pid'].astype(int)):
    train = train_data.iloc[train_index].reset_index(drop=True)  # create training dataframe with indices from fold split
    valid = train_data.iloc[val_index].reset_index(drop=True)
    print('train:', len(train))
    train.to_csv("/home/doju/data/chest14/gender/F_" + str(int(p*100)) + "/train_fold" + str(fold_no) + ".csv")
    print('val:', len(valid))
    valid.to_csv("/home/doju/data/chest14/gender/F_" + str(int(p*100)) +"/val_fold" + str(fold_no) + ".csv")
    fold_no += 1

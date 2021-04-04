import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import pickle
from sklearn.ensemble import RandomForestClassifier
mpl.rcParams['patch.force_edgecolor'] = True
# import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

training_data = pd.read_csv('training_data_2_csv_UTF.csv')
bots = training_data[training_data.bot==1]
nonbots = training_data[training_data.bot==0]

condition = (bots.screen_name.str.contains("bot", case=False)==True)|(bots.description.str.contains("bot", case=False)==True)|(bots.location.isnull())|(bots.verified==False)

bots['screen_name_binary'] = (bots.screen_name.str.contains("bot", case=False)==True)
bots['location_binary'] = (bots.location.isnull())
bots['verified_binary'] = (bots.verified==False)
# bots.shape

condition1 = (nonbots.screen_name.str.contains("bot", case=False)==False)| (nonbots.description.str.contains("bot", case=False)==False) |(nonbots.location.isnull()==False)|(nonbots.verified==True)

nonbots['screen_name_binary'] = (nonbots.screen_name.str.contains("bot", case=False)==False)
nonbots['location_binary'] = (nonbots.location.isnull()==False)
nonbots['verified_binary'] = (nonbots.verified==True)

# nonbots.shape

df = pd.concat([bots, nonbots])
df.corr(method='spearman')

training_data = pd.read_csv('training_data_2_csv_UTF.csv')

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
            
training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)

training_data['listed_count_binary'] = (training_data.listed_count>20000)==False
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]

dt = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dt = dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))
# print(dt.predict([[True, True, True, False, False, 1129, 7, 23557, True]]))
# print(dt.predict([[False, False, False, False, True, 571310, 76070, 56077, True]]))

out1 =dt.predict([[True, True, True, False, False, 1129, 7, 23557, True]])

if int(out1) == 1: 
    print('A BOT')
else: 
    print('NOT A BOT')

out2 =dt.predict([[False, False, False, False, True, 571310, 76070, 56077, True]])

if int(out2) == 1: 
    print('A BOT')
else: 
    print('NOT A BOT')    

pickle.dump(dt, open('model3.pkl','wb'))    
import numpy as np
import tweepy
from pandas import DataFrame   
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
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

# Building the model
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




# Create your dictionary class
class my_dictionary(dict):

	# __init__ function
	def __init__(self):
		self = dict()
		
	# Function to add key:value
	def add(self, key, state, name, profile_image, url):
		self[key] = [state, name, profile_image, url]




#Initializing the flask App
app = Flask(__name__) #Initialize the flask App
# model = pickle.load(open('model3.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index (1).html')

@app.route('/faq')
def faq():
    return render_template('FAQ.html')

@app.route('/explore')
def explore():
    return render_template('Explore.html')    

@app.route('/teams')
def teams():
    return render_template('Teams.html')        

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    f = request.files['screen_name_binary_file']
    input_user = request.form['screen_name_binary']

    consumer_key="MBTx7P4qTWcTuagfBhnuujglQ"
    consumer_secret="d3Ahs3B7Gl8vNv2S7PY97zoXFLX0LAqEf37DJAnygI7WqdknYS"
    access_token="3182828562-pdnEkOmRkY25fPPyUnjmhemgOl41rek26ucJrub"
    access_token_secret="SzU7hJMCcxYXVPTrdsdCy6kEWZvEtK4HA4cS8f2VjKr9I"

    # authorization of consumer key and consumer secret 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    # set access to user's access key and access secret  
    auth.set_access_token(access_token, access_token_secret)
    
    # calling the api  
    api = tweepy.API(auth) 

    # the screen name of the user
    # content = f.read()

    # Screen_name_binary = content.split(b',')

    # screen_name = (Screen_name_binary[0])

    # # fetching the user 
    # user = api.get_user(screen_name)

    # name = user.name
    # description = user.description
    # screen_name = user.screen_name
    # status = user.status

    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

    # list1 = [name, description, screen_name, status]

    # df = DataFrame(list1, columns = ['binary_features'])

    # df['values'] = df.binary_features.str.contains(bag_of_words_bot, case=False, na=False)

    # Screen_name_binary= df.values[2][1]                
    # Name_binary = df.values[0][1]
    # Description_binary = df.values[1][1]
    # Status_binary = df.values[3][1]
    # Verified = user.verified
    # Followers_count = user.followers_count
    # Friends_count = user.friends_count
    # Statuses_count = user.statuses_count
    # Listed_count_binary = (user.listed_count>20000)==False

    # result = model.predict([[Screen_name_binary, Name_binary, Description_binary, Status_binary, Verified, Followers_count, Friends_count, Statuses_count, Listed_count_binary]])
    # output = result[0]
    # if int(output) == 1: 
    #     prediction = 'THE PROVIDED ACCOUNT IS A BOT.'
    # else: 
    #     prediction = 'THE PROVIDED ACCOUNT IS NOT A BOT.'
    # # output = prediction[0]
    # return render_template('predict.html', prediction_text = prediction, Username = screen_name)
    results = my_dictionary()
    # for screen_name in Screen_name_binary:
    #     user = api.get_user(screen_name)
    #     name = user.name
    #     description = user.description
    #     screen_name = user.screen_name
    #     status = user.status
    #     list1 = [name, description, screen_name, status]
    #     df = DataFrame(list1, columns = ['binary_features'])
    #     df['values'] = df.binary_features.str.contains(bag_of_words_bot, case=False, na=False)
    #     Screen_name_binary= df.values[2][1]                
    #     Name_binary = df.values[0][1]
    #     Description_binary = df.values[1][1]
    #     Status_binary = df.values[3][1]
    #     Verified = user.verified
    #     Followers_count = user.followers_count
    #     Friends_count = user.friends_count
    #     Statuses_count = user.statuses_count
    #     Listed_count_binary = (user.listed_count>20000)==False
    #     profile_image = user.profile_image_url
    #     url = "https://twitter.com/"+screen_name
    #     print(url)
    #     result = model.predict([[Screen_name_binary, Name_binary, Description_binary, Status_binary, Verified, Followers_count, Friends_count, Statuses_count, Listed_count_binary]])
    #     output = result[0]

    #     if int(output) == 1: 
    #         # prediction = 'THE PROVIDED ACCOUNT IS A BOT.'
    #         results.add(screen_name, 'BOT', profile_image, url)
    #     else: 
    #         # prediction = 'THE PROVIDED ACCOUNT IS NOT A BOT.'
    #         results.add(screen_name, 'NOT A BOT', profile_image, url)

    # # return render_template('predict.html', prediction_text = prediction, Username = screen_name)
    # return render_template('predict.html', results = results)      
    #when only one name is submitted and not the file
    if input_user and not f:
        screen_name = input_user
        user = api.get_user(screen_name)
        name = user.name
        description = user.description
        screen_name = user.screen_name
        status = user.status
        list1 = [name, description, screen_name, status]
        df = DataFrame(list1, columns = ['binary_features'])
        df['values'] = df.binary_features.str.contains(bag_of_words_bot, case=False, na=False)
        Screen_name_binary= df.values[2][1]                
        Name_binary = df.values[0][1]
        Description_binary = df.values[1][1]
        Status_binary = df.values[3][1]
        Verified = user.verified
        Followers_count = user.followers_count
        Friends_count = user.friends_count
        Statuses_count = user.statuses_count
        Listed_count_binary = (user.listed_count>20000)==False
        profile_image = user.profile_image_url
        url = "https://twitter.com/"+screen_name
        
        # print(url)
        result = dt.predict([[Screen_name_binary, Name_binary, Description_binary, Status_binary, Verified, Followers_count, Friends_count, Statuses_count, Listed_count_binary]])
        output = result[0]

        if int(output) == 1: 
            # prediction = 'THE PROVIDED ACCOUNT IS A BOT.'
            results.add(screen_name, 'BOT', name, profile_image, url)
        else: 
            # prediction = 'THE PROVIDED ACCOUNT IS NOT A BOT.'
            results.add(screen_name, 'NOT A BOT', name, profile_image, url)
        return render_template('predict.html', results = results)
    elif f and not input_user:
        content = f.read()
        Screen_name_binary = content.split(b',')
        for screen_name in Screen_name_binary:
            user = api.get_user(screen_name)
            name = user.name
            description = user.description
            screen_name = user.screen_name
            status = user.status
            list1 = [name, description, screen_name, status]
            df = DataFrame(list1, columns = ['binary_features'])
            df['values'] = df.binary_features.str.contains(bag_of_words_bot, case=False, na=False)
            Screen_name_binary= df.values[2][1]                
            Name_binary = df.values[0][1]
            Description_binary = df.values[1][1]
            Status_binary = df.values[3][1]
            Verified = user.verified
            Followers_count = user.followers_count
            Friends_count = user.friends_count
            Statuses_count = user.statuses_count
            Listed_count_binary = (user.listed_count>20000)==False
            profile_image = user.profile_image_url
            url = "https://twitter.com/"+screen_name
            
            # print(url)
            result = dt.predict([[Screen_name_binary, Name_binary, Description_binary, Status_binary, Verified, Followers_count, Friends_count, Statuses_count, Listed_count_binary]])
            output = result[0]

            if int(output) == 1: 
                # prediction = 'THE PROVIDED ACCOUNT IS A BOT.'
                results.add(screen_name, 'BOT', name, profile_image, url)
            else: 
                # prediction = 'THE PROVIDED ACCOUNT IS NOT A BOT.'
                results.add(screen_name, 'NOT A BOT', name, profile_image, url)
        return render_template('predict.html', results = results)
    elif input_user and f:
        message = "YOU CAN ONLY USE ONE AT A TIME"
        return render_template('index (1).html', message = message)
    else:
        message = "ENTER THE ACCOUNT INFORMATION PLEASE"
        return render_template('index (1).html', message = message)  

  


if __name__ == "__main__":
    app.run(debug=True)

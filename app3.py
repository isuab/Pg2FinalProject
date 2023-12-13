import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm 


## Initialize variables
# age_in = 1000
gender_in = 'Female'
income_in = 'Less than $10,000'
educ2_in = 'Less than high school (Grades 1-8 or no formal schooling'
marital_in = 'Married'
par_in = 'Children'


st.write("Predicting LinkedIn User Status")
#Ask for user inputs for predictions

# Age as a slider:
age_in = st.slider(
    'Select a range of values',
    0, 97, (0, 50)
)
age_con = age_in[1]

## Add drop down select boxes
#gender
gender_in = st.selectbox(
    'Gender:',
    ('Male', 'Female'))
if gender_in == 'Male':
	gender_con = 1
else:
	gender_con = 0   # female

#income
income_in = st.selectbox(
    'Income Bracket:',
    ('Less than $10,000', '10 to under $20,000', '30 to under $40,000', '40 to under $50,000', '50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'))

if income_in == 'Less than $10,000':
	income_con = 1
elif income_in == '10 to under $20,000':
	income_con = 2
elif income_in == '20 to under $30,000':
	income_con = 3
elif income_in == '30 to under $40,000':
	income_con = 4
elif income_in == '40 to under $50,000':
	income_con = 5
elif income_in == '50 to under $75,000':
	income_con = 6
elif income_in == '75 to under $100,000':
	income_con = 7
elif income_in == '100 to under $150,000':
	income_con = 8
else:
	income_con = 9		# $150,000 or more

# education
educ2_in = st.selectbox(
    'Education Bracket:',
    ('Less than high school (Grades 1-8 or no formal schooling)', 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)', 'High school graduate (Grade 12 with diploma or GED certificate)', 'Some college, no degree (includes some community college)', 'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB', 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)', 'Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'))

if educ2_in == 'Less than high school (Grades 1-8 or no formal schooling':
	educ2_con = 1
elif educ2_in == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
	educ2_con = 2
elif educ2_in == 'High school graduate (Grade 12 with diploma or GED certificate)':
	educ2_con = 3
elif educ2_in == 'Some college, no degree (includes some community college)':
	educ2_con = 4
elif educ2_in == 'Two-year associate degree from a college or university':
	educ2_con = 5
elif educ2_in == 'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)':
	educ2_con = 6
elif educ2_in == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
	educ2_con = 7
else:
	educ2_con = 8		# Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)

#marital
marital_in = st.selectbox(
    'Marital Status:',
    ('Married', 'Not Married (Single, Divorced, Widowed, Partnered, Etc.)'))

if marital_in == 'Married':
	marital_con = 1
else:
	marital_con = 0

#parent
par_in = st.selectbox(
    'Parent Status:',
    ('Children', 'No Children'))

if par_in == 'Children':
	par_con = 1
else:
	par_con = 0


## ADD SUBMIT BUTTON
submit_button = st.button(label='Submit')


#read in the data
##s = pd.read_csv('C:\\Users\\isuab\\Downloads\\social_media_usage.csv')
s = pd.read_csv('social_media_usage.csv')

#define clean up function
def clean_sm(x):
# x can be either a dataframe or a column
    new_x = np.where(x==1,1,0)
    return new_x
#set data frame with variables of interest 
ss = s[['web1h','income','educ2','par','marital','gender','age']]
#make anything in the income column above 9, nan
ss.loc[ss['income']>9,'income']=np.nan
#make anything in the education column above 8, nan
ss.loc[ss['educ2']>8,'educ2']=np.nan
#make anything in the age column above 97, nan
ss.loc[ss['age']>97,'age']=np.nan
#clean up marital status, first make anything above 6 nan and then apply the clean_sm function
ss.loc[ss['marital']>6,'marital']=np.nan
#clean up gender, first make anything above 2 nan and then apply the clean_sm function
ss.loc[ss['gender']>2,'gender']=np.nan
#clean up parental, first make anything above 2 nan and then apply the clean_sm function
ss.loc[ss['par']>2,'par']=np.nan
#clean up Linkedin, first make anything above 2 nan and then apply the clean_sm function
ss.loc[ss['web1h']>2,'web1h']=np.nan
#step 2
ss=ss.dropna()
#apply the clean_sm function
ssbinary=ss[['web1h','par','marital','gender']]
ssbinary=ssbinary.applymap(clean_sm)
#replace the binary columns in the ss dataframe
ss['web1h']=ssbinary['web1h']
ss['par']=ssbinary['par']
ss['marital']=ssbinary['marital']
ss['gender']=ssbinary['gender']
#rename target variable
ss=ss.rename(columns={'web1h':'sm_li'})

#Model Fitting 
#target 
y=ss['sm_li']
#features
X=ss[['income','marital','gender','age','par','educ2']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)
mod=LogisticRegression(class_weight='balanced')
mod.fit(X_train,y_train)


## Predictions Area
def make_predictions(income_con, gender_con, age_con, marital_con, par_con, educ2_con):
	new = {'income': income_con, 'marital': marital_con,'gender': gender_con, 'age': age_con,'par': par_con,'educ2': educ2_con}
	new = pd.DataFrame([new])
	pred = mod.predict(new)
	
	prob = mod.predict_proba(new)

	return prob



if submit_button == True:
	response = make_predictions(income_con, gender_con, age_con, marital_con, par_con, educ2_con)
	st.write(response)
	
	if response[0,1] > 0.5:
		st.write('The person is a LinkedIn User')
	else:
		st.write('The person is NOT a LinkedIn User')










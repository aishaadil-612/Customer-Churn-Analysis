import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#df= pd.DataFrame("customer_churn_dataset-testing-master.csv")
df=pd.read_csv("customer_churn_dataset-testing-master.csv")
# cleaning of data
shape =df.shape
print("the shape is",shape)
print(df.info())
print(df.head())
print( df.describe())
df.fillna(0,inplace =True)
df.drop_duplicates()
for col in df.select_dtypes(include='object'):
    df[col] = df[col].str.strip()
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.title()   
df.replace('', np.nan, inplace=True)
print(df.isnull().sum())
print(df['Churn'].value_counts())

# churn distribution
x= df['Churn'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(x.index,x.values, color='orange',edgecolor='black')
plt.title("churn distribution")
plt.xlabel('index') 
plt.ylabel('vale') 
plt.savefig("churndistribution.png")
plt.show()

churn_rate = df['Churn'].mean() * 100
print(f"Churn Rate: {churn_rate:.2f}%")
pd.crosstab(df['Gender'], df['Churn'])
churn_counts = df['Churn'].value_counts()
# churn no churn
plt.figure(figsize=(6,6))
plt.pie(
    churn_counts.values,
    labels=['No Churn', 'Churn'],
    autopct='%1.1f%%'
)
plt.title('Customer Churn Percentage')
plt.savefig("churn.png")

plt.show()
# contract vs churn

contract_churn = pd.crosstab(
    df['Contract Length'],
    df['Churn']
)

contract_churn.plot(
    kind='bar',
    figsize=(8,5)
)

plt.title('Contract Type vs Churn')
plt.xlabel('Contract Type')
plt.ylabel('Customer Count')
plt.xticks(rotation=0)
plt.savefig("contractvschurn.png")
plt.show()
# totalspend distribution

plt.figure(figsize=(8,5))

plt.hist(
    df['Total Spend'],
    bins=20
)

plt.title('total Distribution')
plt.xlabel('total spend')
plt.ylabel('Frequency')
plt.savefig("totaldistribution.png")

plt.show()
# custumer tenure distribution
plt.figure(figsize=(8,5))

plt.hist(
    df['Tenure'],
    bins=20
)

plt.title('Customer Tenure Distribution')
plt.xlabel('Tenure (Months)')
plt.ylabel('Customers')
plt.savefig("custumertenure.png")

plt.show()
# monthly charges vs churn

plt.figure(figsize=(8,5))
# monthly charges vs churn
df.boxplot(
    column='Total Spend',
    by='Churn'
)

plt.title('Monthly Charges by Churn')
plt.suptitle('')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')

plt.savefig("monthlybychurn.png")

plt.show()
# tenure by churn

plt.figure(figsize=(8,5))

df.boxplot(
    column='Tenure',
    by='Churn'
)


plt.title('Tenure by Churn')
plt.suptitle('')
plt.xlabel('Churn')
plt.ylabel('Tenure')
plt.savefig("tenurebychurn.png")
plt.show()


# . Subscription Type vs churn
payment_churn = pd.crosstab(
    df['Subscription Type'],
    df['Churn']
)

payment_churn.plot(
    kind='bar',
    figsize=(10,5)
)

plt.title('Subscription Type vs Churn')
plt.xlabel('Subscription Type')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)
plt.savefig("subscriptionbychurn.png")
plt.show()

age_by_churn = df.groupby('Churn')['Age'].mean()

plt.pie(
    age_by_churn.values,
    labels=age_by_churn.index,
    autopct='%1.1f%%'
)

plt.title("Average Age by Churn Status")
plt.savefig("agebychurn.png")
plt.show()
# customer segmentTation using numpy 
conditions = [
    df['Tenure'] <= 12,
    (df['Tenure'] > 12) & (df['Tenure'] <= 36),
    df['Tenure'] > 36
]

choices = [
    'New Customer',
    'Regular Customer',
    'Loyal Customer'
]

df['customer_type'] = np.select(
    conditions,
    choices,
    default='Unknown'
)
print(df['customer_type'])
df['risk_score'] = np.where(
    (df['Tenure'] < 12) &
    (df['Total Spend'] > 70),
    'High Risk',
    'Low Risk'
)
print(df['risk_score'])
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
plt.imshow(corr, aspect='auto')
plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title('Correlation Matrix')
plt.savefig("correlationmatrix.png")
plt.show()
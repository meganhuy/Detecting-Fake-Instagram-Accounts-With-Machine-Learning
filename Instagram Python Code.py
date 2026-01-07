# %% [markdown]
# Name: Megan Huy
# 
# Project: Detect Fake and Real Instagram Accounts

# %% [markdown]
# ## Objective ##
# -Develop an ML model to classify Instagram accounts as fake or real.
# 
# -Optimize for accuracy, sensitivity, and specificity while reducing cost.
# 
# -Deploy the model for scalability and cost efficiency.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import statsmodels.formula.api as smf

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## Instagram Data ##

# %%
#Import Train and Test Data
train_data = pd.read_csv("C:/Users/Megan Huy/Documents/Instagram Project/instagram-data-train.csv") 
test_data = pd.read_csv("C:/Users/Megan Huy/Documents/Instagram Project/instagram-data-test.csv")

# %%
#Combine Train and Test Data
combined = pd.concat([train_data, test_data], ignore_index=True)
print(combined.head())
print(combined.shape)

# %% [markdown]
# ## EDA ##

# %%
#Check Data Types
combined.info()

# %%
#Binary vs Numerical
numeric_cols = combined.select_dtypes(include=["int64", "float64"]).columns
binary_cols = [col for col in numeric_cols if train_data[col].nunique() == 2]
other_numeric_cols = [col for col in numeric_cols if col not in binary_cols]

print("Binary columns:", binary_cols)
print("Other numeric columns:", other_numeric_cols)

# %%
#Rename Variables
instagram_data = combined.rename(columns={
    "profile pic": "profile_pic",
    "nums/length username": "username_with_numbers",
    "fullname words": "wordcount_fullname",
    "nums/length fullname": "fullname_with_numbers",
    "name==username": "fullname_is_username",
    "description length": "bio_length",
    "external URL": "external_url",
    "#posts": "posts",
    "#followers": "followers",
    "#follows": "follows",
    "fake": "fake",          
    "private": "private",    
    "dataset": "dataset"
})
print(instagram_data.head())

# %%
#Check For Missing Values
missing_counts = instagram_data.isna().sum()
print(missing_counts)

#Percent missing per column
missing_percent2 = instagram_data.isna().mean() * 100
print(missing_percent2)

# %% [markdown]
# There are no missing values in the dataset.

# %%
#Percentages
percentages = (
    instagram_data
    .groupby('fake')
    .size()
    .reset_index(name='n')
)

#Add percentage
percentages['percent'] = percentages['n'] / percentages['n'].sum() * 100
#Add label
percentages['label'] = percentages['fake'].map({1: 'Fake', 0: 'Real'})
print(percentages)

# %%
#Bar Plot
x = ['Accounts']
real_percent = percentages.loc[percentages['label'] == 'Real', 'percent'].values[0]
fake_percent = percentages.loc[percentages['label'] == 'Fake', 'percent'].values[0]

fig, ax = plt.subplots()

#Bottom = Real
ax.bar(x, [real_percent], label='Real', color='purple', edgecolor='black', width=0.6)
# Top = Fake
ax.bar(x, [fake_percent], bottom=[real_percent], label='Fake', color='hotpink', edgecolor='black', width=0.6)

#Add Text Labels
ax.text(0, real_percent / 2, f"{real_percent:.1f}%", ha='center', va='center', color='white')
ax.text(0, real_percent + fake_percent / 2, f"{fake_percent:.1f}%", ha='center', va='center', color='white')
ax.set_ylabel("Percentage")
ax.set_title("Percent Proportion of Fake vs Real Accounts")
ax.set_xticks([])  
ax.legend(title="Account Type")

#Plot Display
plt.tight_layout()
plt.show()

# %% [markdown]
# There's an equal distribution of fake accounts and real accounts.

# %%
#Summary Statistics
print(instagram_data.describe(include="all"))

# %%
#Selected Columns for Histograms
cols = [
    "username_with_numbers",
    "wordcount_fullname",
    "bio_length",
    "fullname_with_numbers",
    "followers",
    "follows",
    "posts",
]

#Longer Format
numeric_long = instagram_data[cols].melt(
    var_name="variable",
    value_name="value"
)

#Plot Histograms
sns.set_theme(style="whitegrid")

g = sns.FacetGrid(
    numeric_long,
    col="variable",
    col_wrap=2,
    sharex=False,
    sharey=False
)
g.map(plt.hist, "value", bins=30, edgecolor="white", color="purple")

g.figure.suptitle("Histograms of Numeric Variables", y=1.02)
g.set_xlabels("")
g.set_ylabels("Frequency")

plt.show()

# %% [markdown]
# Most numerical variables are heavily-skewed.

# %%
features_df = instagram_data.select_dtypes(include=[np.number])

#Compute correlation matrix 
cor_matrix = features_df.corr(method="pearson")
colors = ["lightyellow", "pink", "purple"]
cmap = LinearSegmentedColormap.from_list("yellow_pink_purple", colors)

#Plot correlation heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(
    cor_matrix,
    annot=True,          
    fmt=".2f",
    cmap=cmap,           
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix Between Features", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# There is a moderately positive correlation between fake accounts and username with numbers. There is a moderately negative correlation between fake accounts and profile pictures.

# %%
#Define binary features
binary_features = ["username_with_numbers", "external_url", "private", "profile_pic", "fullname_is_username"]

#Prepare long data
df = instagram_data.copy()

#Fake-Real Labels (1/0)
df["label"] = df["fake"].map({1: "Fake", 0: "Real"})

#Long Format
long_data = df.melt(
    id_vars=["label"],
    value_vars=binary_features,
    var_name="feature",
    value_name="value"
)

# Keep only rows where value == 1
long_data = long_data[long_data["value"] == 1]

#Rename
feature_labels = {
    "username_with_numbers": "Username has Numbers",
    "external_url": "External URL",
    "private": "Private Account",
    "profile_pic": "Profile Picture",
    "fullname_is_username": "Full Name is the Username"
}
long_data["feature"] = long_data["feature"].map(feature_labels)

#Count and compute percentage
plot_data = (
    long_data
    .groupby(["feature", "label"], as_index=False)
    .size()
    .rename(columns={"size": "n"})
)

#Percent within each feature
plot_data["percent"] = (
    plot_data.groupby("feature")["n"]
    .transform(lambda x: x / x.sum() * 100)
)

#Pivot for stacked bar plot
pivot = plot_data.pivot(
    index="feature",
    columns="label",
    values="percent"
).fillna(0)

pivot = pivot[["Real", "Fake"]]  

#Plot stacked percent bar chart
ax = pivot.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color={"Real": "purple", "Fake": "hotpink"},
    edgecolor="black"
)

ax.set_title("Distribution of Real vs Fake Accounts with Binary Variables")
ax.set_xlabel("Features")
ax.set_ylabel("Percentage")

#Rotate x-axis labels
plt.xticks(rotation=30, ha="right")

#Format y-axis as percentage (0–100 with %)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

#Add percentage labels inside bars
for container in ax.containers:
    # For each segment in the stacked bars
    for bar in container:
        height = bar.get_height()
        if height > 0:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_y() + height / 2
            ax.text(
                x, y,
                f"{height:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=9
            )

plt.tight_layout()
plt.show()

# %% [markdown]
# There were no fake accounts with external URL. There were more fake accounts with full name as their username and less fake accounts were set to private or had a picture in their profile.

# %% [markdown]
# ## Logistic Regression ##

# %%
#Full Logistic Regression Model
full_model = smf.logit(
    formula="fake ~ profile_pic + fullname_is_username + external_url + private + "
            "username_with_numbers + fullname_with_numbers + wordcount_fullname + "
            "bio_length + posts + followers + follows",
    data=instagram_data
).fit()

print(full_model.summary())

# %%
#Odd Ratios for Full Model
ci = full_model.conf_int()  
ci_exp = np.exp(ci)
table = or_table = pd.DataFrame({
    "OR": np.exp(full_model.params),
    "Lower CI": ci_exp[0],
    "Upper CI": ci_exp[1],
    "p-value": full_model.pvalues
})

print(table)

# %% [markdown]
# The likelihood of fake accounts is with features that have odds ratio > 1. This includes four features: full names with numbers, full name as the username, username with numbers, and follows. 
# 
# There is strong effect but variability is large for full name as username, full name with numbers, and username with numbers. There is a small effect for accounts with follows.
# 
# Using a significance level of α = 0.05, the model shows that profile picture, privacy, username with numbers, bio length, posts, followers, and followed accounts are all statistically significant predictors of the outcome.

# %%
#Gini Importance using Random Forest
X = instagram_data.drop(columns="fake")   
y = instagram_data["fake"]           

rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=123
)
rf_model.fit(X, y)

var_imp_df = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print(var_imp_df)

# %% [markdown]
# We chose the five, top predictors for the reduced logistic regression model.

# %%
#Plot Variable Importance

plt.figure(figsize=(8, 6))
plt.barh(
    var_imp_df["Variable"],
    var_imp_df["Importance"],
    color="purple"
)
plt.gca().invert_yaxis()
plt.xlabel("Mean Decrease in Gini")
plt.ylabel("Variable")
plt.title("Random Forest Variable Importance")
plt.tight_layout()
plt.show()



# %%
#Reduced Model
reduced_model = smf.logit(
    formula="fake ~ followers + username_with_numbers + posts + profile_pic + bio_length",
    data=instagram_data
).fit()

print(reduced_model.summary())

# %%
#Odds Ratios for Reduced Model
ci = reduced_model.conf_int()  
ci_exp = np.exp(ci)
reduced_table = or_table = pd.DataFrame({
    "OR": np.exp(reduced_model.params),
    "Lower CI": ci_exp[0],
    "Upper CI": ci_exp[1],
    "p-value": reduced_model.pvalues
})

print(reduced_table)

# %%
#AIC Comparison
print("Full Model AIC:", full_model.aic)
print("Reduced Model AIC:", reduced_model.aic)


# %% [markdown]
# Given AIC, we decided to keep the full model and use it for interpretation. 
# The full model had a substantially lower AIC than the selected model, indicating that it provided a better overall fit to the data despite having more predictors.
# 
# We retained the full model for interpretation, as it captured more of the relevant relationships in the data than the reduced model. We will examine the odds ratios from the full model to quantify how changes in each predictor affected the odds of an account being classified as Fake, identifying which features increases those odds.

# %% [markdown]
# ## Predictive Models ##

# %%
#Restore Data
train_data = pd.read_csv(r"C:\Users\Megan Huy\Documents\Instagram Project\instagram-data-train.csv")
test_data  = pd.read_csv(r"C:\Users\Megan Huy\Documents\Instagram Project\instagram-data-test.csv")

#Join Train and Test 
train_data["dataset"] = "train"
test_data["dataset"]  = "test"
instagram_data2 = pd.concat([train_data, test_data], ignore_index=True)

#Rename Variables
rename_map = {
    "profile pic": "profile_pic",
    "nums/length username": "username_with_numbers",
    "fullname words": "wordcount_fullname",
    "nums/length fullname": "fullname_with_numbers",
    "name==username": "fullname_is_username",
    "description length": "bio_length",
    "external URL": "external_url",
    "#posts": "posts",
    "#followers": "followers",
    "#follows": "follows",
}
instagram_data2 = instagram_data2.rename(columns=rename_map)

#Create Fake Label 0=no, 1=yes
instagram_data2["fake"] = instagram_data2["fake"].map({0: "no", 1: "yes"})

#Select Variables
model_vars = ["fake", "followers", "fullname_is_username", "username_with_numbers",
              "fullname_with_numbers", "follows", "dataset"]

instagram_data2 = instagram_data2[model_vars].dropna()

#Check if Fake Variable and Label are Correct
print(instagram_data2["fake"].value_counts(dropna=False))

# %%
#Train-Test Split
train_set, test_set = train_test_split(
    instagram_data2,
    test_size=0.20,        
    random_state=123,   
    stratify=instagram_data2["fake"]  
)

print("\nTest set class counts:")
print(test_set["fake"].value_counts())


# %%

#Prepare Data for Modeling
X_train = train_set.drop(columns=["fake", "dataset"])
y_train = train_set["fake"].map({"no": 0, "yes": 1})
X_test  = test_set.drop(columns=["fake", "dataset"])
y_test  = test_set["fake"].map({"no": 0, "yes": 1})


#Cross-validation Setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)


#Models (RF, KNN with Scaling, Logistic)
rf_model = RandomForestClassifier(n_estimators=500, random_state=123)

knn_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

log_model = LogisticRegression(max_iter=2000, solver="lbfgs")

#ROC-AUC Curve on Train Set
rf_cv_auc  = cross_val_score(rf_model,  X_train, y_train, cv=cv, scoring="roc_auc").mean()
knn_cv_auc = cross_val_score(knn_model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
log_cv_auc = cross_val_score(log_model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

print("10-fold CV ROC-AUC (train only):")
print(f"  Random Forest: {rf_cv_auc:.3f}")
print(f"  KNN:          {knn_cv_auc:.3f}")
print(f"  Logistic:     {log_cv_auc:.3f}")


#Fit on Train, evaluate on Test
models = {
    "Random Forest": rf_model,
    "KNN": knn_model,
    "Logistic": log_model
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "cm": cm,
        "f1": f1,
        "auc": auc,
        "y_proba": y_proba
    }

    print("\n" + "="*60)
    print(name)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
    print(f"F1 (positive=1): {f1:.3f}")
    print(f"AUC: {auc:.3f}")


#ROC Curve on Test Set
plt.figure()

for name, out in results.items():
    fpr, tpr, _ = roc_curve(y_test, out["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={out['auc']:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set")
plt.legend(loc="lower right")
plt.show()


#Print AUCs
print("\nAUC - Random Forest:", results["Random Forest"]["auc"])
print("AUC - Logistic:", results["Logistic"]["auc"])
print("AUC - KNN:", results["KNN"]["auc"])


# %% [markdown]
# Random forest has the best AUC score of 0.957.



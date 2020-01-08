import pandas as pd
from matplotlib import pyplot as plt
#in jupyter notebook: %matplotlib inline
import seaborn as sns

df = pd.read_csv('dataset\Pokemon.csv', encoding = "ISO-8859-1", index_col=0)
df.head()

#%% Scatter plot
sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=False, #No regression line as no dedicated scatterplot funct
           hue='Stage') #color by evolution stage
#tweak using matplotlib
plt.ylim(0, None)
plt.xlim(0, None)

sns.boxplot(data=df)

stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
sns.boxplot(data=stats_df)

sns.set_style('whitegrid')
sns.violinplot(x='Type1', y='Attack', data=df)
#recolor
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]
sns.violinplot(x='Type1', y='Attack', data=df, palette=pkmn_type_colors)
sns.swarmplot(x='Type 1', y='Attack', data=df, palette=pkmn_type_colors)

#overlayingplots
plt.figure(figsize=(10,6))
sns.violinplot(x='Type1', y='Attack', data=df,
               inner=None, #remove bars inside violin
               palette=pkmn_type_colors)
sns.swarmplot(x='Type1', y='Attack', data=df, 
              color='k', alpha=0.7) #alpha=>low transparency
plt.title('Attack by type')
stats_df.head()
melted_df=pd.melt(stats_df,
                  id_vars=["Name","Type 1", "Type 2"], #features that won't combine
                  var_name="Stat") #name of new melted var
melted_df.head()
print(stats_df.shape)
print(melted_df.shape) #rwos*6

# Swarmplot with melted_df
sns.swarmplot(x='Stat', y='value', data=melted_df, 
              hue='Type 1')

# Tweaks:  1. Enlarge the plot
plt.figure(figsize=(10,6))
 
sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type 1', 
              split=True, # 2. Separate points by hue
              palette=pkmn_type_colors) # 3. Use Pokemon palette
 
# 4. Adjust the y-axis
plt.ylim(0, 260)
 
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)

# Calculate correlations
corr = stats_df.corr()
# Heatmap
sns.heatmap(corr,annot=True)
# Distribution Plot (a.k.a. Histogram)
sns.distplot(df.Attack)
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)
# Rotate x-labels
plt.xticks(rotation=-45)

# Factor Plot
g = sns.factorplot(x='Type 1', 
                   y='Attack', 
                   data=df, 
                   hue='Stage',  # Color by stage
                   col='Stage',  # Separate by stage
                   kind='swarm') # Swarmplot
 
# Rotate x-axis labels
g.set_xticklabels(rotation=-45)
# Doesn't work because only rotates last plot
# plt.xticks(rotation=-45)

# Density Plot
sns.kdeplot(df.Attack, df.Defense)
# Joint Distribution Plot
sns.jointplot(x='Attack', y='Defense', data=df)
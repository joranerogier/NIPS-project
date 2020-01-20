import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
from scipy import stats

path_first_model = "output/reward_csv/first_model.csv"
path_second_model = "output/reward_csv/model_extra.csv" #
path_third_model = "output/reward_csv/GameResults.csv" # Rodrigo's model
path_fourth_model = "output/reward_csv/GameResults2.csv" # Rodrigo's model
path_fifth_model =  "output/reward_csv/model_9layers.csv"
path_sixth_model =  "output/reward_csv/model_6layers.csv"

models =  ["Alt. DQN 1", "Alt. DQN 2"] #["3 Layers", "6 Layers", "9 Layers", "Extra"]#
percentage_won = [72, 71] # 88, 72, 71, 77] # on 100 games [83, 76, 77, 88]

sns.set(style="whitegrid", font_scale=6)
plt.figure(figsize=(25, 30))
ax1 = sns.barplot(x=models, y=percentage_won)
ax1.set(ylim=(0,100))
ax1.set(xlabel="Models", ylabel="Percentage games won (%)")#, title="Model Performances Alternative DQNs")
plt.savefig("output/model_performance_rqn.jpg")
#plt.show()

data_model1 = pd.read_csv(path_first_model)
timesteps_model_1 = list(data_model1['timesteps'])
episodes_model_1 = list(data_model1['episode nr'])
results_model_1 = list(data_model1['result'])

data_model2 = pd.read_csv(path_second_model)
timesteps_model_2 = list(data_model2['timesteps'])
episodes_model_2 = list(data_model2['episode nr'])
results_model_2 = list(data_model2['result'])

data_model3 = pd.read_csv(path_third_model)
timesteps_model_3 = list(data_model3['timesteps'])
episodes_model_3 = list(data_model3['episode nr'])
results_model_3 = list(data_model3['result'])

data_model4 = pd.read_csv(path_fourth_model)
timesteps_model_4 = list(data_model4['timesteps'])
episodes_model_4 = list(data_model4['episode nr'])
results_model_4 = list(data_model4['result'])

data_model5 = pd.read_csv(path_fifth_model)
timesteps_model_5 = list(data_model5['timesteps'])
episodes_model_5 = list(data_model5['episode nr'])
results_model_5 = list(data_model5['result'])

data_model6 = pd.read_csv(path_sixth_model)
timesteps_model_6 = list(data_model6['timesteps'])
episodes_model_6 = list(data_model6['episode nr'])
results_model_6 = list(data_model6['result'])

timesteps_models = [timesteps_model_3, timesteps_model_4]#, timesteps_model_5, timesteps_model_2]#, timesteps_model_4, timesteps_model_5]
results_models = [results_model_3, results_model_4] #, results_model_5, results_model_2]#, results_model_4, results_model_5]
model_track = []
episode_track = []
timesteps_together = []
results_together = []

model_count = 0
# create dataframe with column of timesteps, episode nr, and model name
for i in timesteps_models:
    c = 1
    for j in i:
        timesteps_together.append(j)
        model_track.append(model_count) # model nr
        episode_track.append(c)
        c += 1
    model_count += 1

for i in results_models:
    for j in i:
        results_together.append(j)


all_tracks = list(zip(timesteps_together, episode_track, model_track, results_together))
#  print(np.array([timesteps_together, episode_track, model_track]))
d = pd.DataFrame(all_tracks, columns=["timesteps","episode_nr", "model", "result"])

# calculate z-value with goal to remove outliers
z = np.abs(stats.zscore(d))
d = d[(z < 3).all(axis=1)] # remove the outliers

model_names = []
for mod in d["model"]:
    mod_name = models[mod]
    model_names.append(mod_name)
d["model"] = model_names


sns.set(style="whitegrid", palette="muted", color_codes=True, font_scale=6)
#plt.figure(figsize=(100, 8))
#fig.set_size_inches(30,30)
ax2 = sns.catplot(x="model", y="timesteps", kind="box", data=d, hue="result", height=25, aspect=1)
#ax2.set(title="Average Timesteps Standard DQN model")
legend_title = "Game Result"
legend_labels = ["Lost", "Won"]
ax2._legend.set_title(legend_title)
for t, l in zip(ax2._legend.texts, legend_labels): t.set_text(l)
ax2.fig.subplots_adjust(top=0.9, right=0.8)
plt.savefig("output/models_timesteps_rqn.jpg")
#plt.show()

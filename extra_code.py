# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:16:59 2021

@author: Isabelle
"""


#%%
#distribution of activity for a stimulus as a function of P_rec
#plot not used


col = {'contra':['stim_pref_hem2stim_hem2', 'stim_pref_hem2stim_hem2'], 'ipsi':['stim_pref_hem1stim_hem2', 'stim_pref_hem2stim_hem1']}
n= 40

indices = {0.08:[], 0.10:[], 0.25:[], 0.50:[], 0.75:[], 1.0:[]}
n_cals = {10:[], 20:[], 30:[], 40:[]}
for i in list(first_set.index):
    if first_set['P_rec'][i] == 0.25:
        indices[first_set['P_rec'][i]].append(i)
        n_cals[first_set['N_cal'][i]].append(i)

figure, ax = plt.subplots(1,1)
figure.suptitle('distribution of activity for a stimulus as a function of P_rec')
x = np.linspace(1, n, n)
for j in list(n_cals.keys()):
    for i in [list(col.keys())[0]]:   
        a = np.array(first_set[col[i]].values.tolist())[n_cals[j]]
        a = np.mean(a, axis=0)
        mean_resp_ord = np.mean(a, axis=0)
        sem = np.std(a, axis=0, ddof=1)/np.sqrt(np.size(a))
        ax.plot(x, mean_resp_ord,label = j)
    
ax.hlines(0, 0, n, colors='grey', linestyles='--', alpha=0.8)
ax.set_xlabel('neuron number')

plt.legend()

#%% 
#plot the relationship between reponse to stim 1 and stim2 for each neurons
unity_line = [-1, 0, 1]

figure = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.scatter(max_hem1_hem1stim, max_hem1_hem2stim, c = 'coral', label = 'hemisphere 1', alpha=0.6)
ax1.scatter(max_hem2_hem1stim, max_hem2_hem2stim, c = 'green', label = 'hemisphere 2', alpha=0.6)
ax1.plot(unity_line, unity_line, c='black')
ax1.set_xlim(-1, 1)
ax1.set_xticks([-1,-0.5,0, 0.5,1])
ax1.set_ylim(-1,1)
ax1.set_yticks([-1,-0.5,0, 0.5,1])
ax1.set_title('states of excitatory neuron in hemisphere 1 and 2 at T = 500 ms')
ax1.legend()
ax1.set_xlabel('stim in hem 1')
ax1.set_ylabel('stim in hem 2')


#%%
#trying to fit a lognormal curve to the distribution of weights
#never used

weight_distrib = np.concatenate(weights['W_rec'][:, 0:nb_excn*2])
weight_distrib = [i for i in weight_distrib if i != 0.0]

stdev = np.std(weight_distrib)
mean = np.mean(weight_distrib)


#just a gaussian fit
fig = plt.figure()
ax = plt.subplot(111)
from scipy.stats import gaussian_kde
density = gaussian_kde(weight_distrib)
xs = np.linspace(0,0.6,150)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax.plot(xs,density(xs))
ax.hist(weight_distrib, bins = 10, density = True)

#the lognormal distrib fitted to the data but does not work
shape, loc, scale = lognorm.fit(weight_distrib, floc = -1)
estimated_mu = np.log(scale)
estimated_sigma = shape

plt.hist(weight_distrib, bins=50, density=True)
xmin = np.min(weight_distrib)
xmax = np.max(weight_distrib)
x = np.linspace(xmin, xmax, 200)
pdf = lognorm.pdf(x, 1.8, scale = estimated_mu)
plt.plot(x, pdf, 'k')
plt.show()

#normal distribution of the logarithms of the weights

log_weights = np.log(weight_distrib)

lmean = np.mean(log_weights)
lstd = np.std(log_weights)
x = np.linspace(-25, 0, 100)
y = norm.pdf(x,lmean,lstd)

plt.hist(log_weights, bins=50, alpha = 0.75, density=True)
plt.plot(x,y, 'k', color='coral')

#%%
#create dictionary to compare the output of the models in response to equal stimuli for different levels of optogenetic stimulation

n_range = [40,80]
s = np.array([0.0, 0.2, 0.4, 0.6])
coherences = {'none_00':{}, 
              'ipsi_25':{}, 'ipsi_50':{}, 'ipsi_75':{}, 'ipsi_100':{},
              'cont_25':{}, 'cont_50':{}, 'cont_75':{}, 'cont_100':{}, 
              'both_25':{}, 'both_50':{}, 'both_75':{}, 'both_100':{}}

s_str = s.astype(str)
for coh in coherences:
    for i in s_str:
        coherences[coh][i] = []


for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    
    dalemodel_test = dict(np.load(f'/UserFolder/neur0003/third_set_models/{item}', allow_pickle=True))
    weights = fcts.adapt_for_opto(dalemodel_test['weights'].reshape(1)[0])
    trials = dalemodel_test['trials'].reshape(1)[0]
    stim_pref_dict = fcts.stim_pref_(trials)
    
    for coh in coherences:
        if coh[0:4]=='cont' or coh[0:4]=='none' or coh[0:4]=='both':
            stim1 = 'max_hem1stim'
            stim2 = 'max_hem2stim'
            
        if coh[0:4]=='ipsi':
            stim1 = 'max_hem2stim'
            stim2 = 'max_hem1stim'
        
        arr1 = stim_pref_dict[stim1][n_range[0]:n_range[1]]
        arr2 = stim_pref_dict[stim2][n_range[0]:n_range[1]]
        indices = fcts.count_pref(arr1, arr2, indices=True)
        if coh[0:4]=='both':
            indices += fcts.count_pref(arr2, arr1, indices=True)
            random.shuffle(indices)
            
        if n_range == [40,80]:
            indices = np.array(indices)
            indices = indices+40
        
        pourc = int(coh[5:])
        indices = indices[:int(pourc/100*len(indices))]
        indices=list(indices)
        weights_modif = fcts.change_opto_stim(weights, indices)
        print(coh, indices)
        
        simulator = BasicSimulator(weights=weights_modif , params = {'dt': 10, 'tau':100})
        
        trials = fcts.gen_pol_trials(simulator, task_pert, [[s[0], s[0]],[s[1],s[1]],[s[2],s[2]],[s[3],s[3]]], sim=True)
        
        for i in range(4):
            coherences[coh][intensity[i-1]].append(trials[f'1:{s[i]}_2:{s[i]}']['model_output'][-1])
            
        #acc = task_pert.accuracy_function(y, model_output, mask)
        #accuracy_opto.append(acc)



#%% 
coherences_hem1_opt = pd.read_pickle('/UserFolder/neur0003/coherences_hem1_dict.pkl')
coherences_hem2_opt = pd.read_pickle('/UserFolder/neur0003/coherences_hem1_dict.pkl')

coherences = {'hem1':coherences_hem1_opt, 'hem2':coherences_hem2_opt}

file_order = [] #15 still in there, but was removed from coherence dictionary because didn't train
for item in os.listdir('/UserFolder/neur0003/third_set_models'):
    file_order.append(item)


coh_averages = {'hem1':np.array([]), 'hem2':np.array([])}
for k in coherences:
    for i in coherences[k]:
        for j in coherences[k][i]:
            coh_averages[k] = np.append(coh_averages[k], np.mean(coherences[k][i][j], axis=0))
    coh_averages[k] = coh_averages[k].reshape(13, 4, 2)    

#%%

Hems = {'hem1':0, 'hem2':1}
Stims = {'ipsi':0, 'contra':1, 'both':2}
hems = list(Hems.keys())
ind = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12]]
plot_data = np.ones((2,3,5,4))
for k in range(len(coherences)):
    for i in range(3):
        plot_data[:,i,:,0] = coh_averages[hems[k]][ind[i],0,0]
        plot_data[:,i,:,1] = np.mean(coh_averages[hems[k]][ind[i],1:,0], axis=1)
        plot_data[:,i,:,2] = coh_averages[hems[k]][ind[i],0,1]
        plot_data[:,i,:,3] = np.mean(coh_averages[hems[k]][ind[i],1:,1], axis=1)
        

stim = 'ipsi'
hem = 'hem1'

labels = ['hem1 no input', 'hem1 input', 'hem2 no input', 'hem2 input']
for i in range(4):
    plt.plot(plot_data[Hems[hem], Stims[stim],:,i], marker='o', label=labels[i])
plt.ylim(-0.05, 1)
plt.xticks([0,1,2,3,4], labels=[0,25,50,75,100])
plt.xlabel('% stimulated cells')
plt.title(f'average output for opto stim of {stim} cells in {hem}')
plt.legend()

#%%
coherence_array = coherences.copy()
for i in coherence_array:
    for j in coherence_array[i]:
        coherence_array[i][j] = np.array(list(coherence_array[i][j].values()))
for i in coherence_array:
    coherence_array[i] = np.array(list(coherence_array[i].values()))
coherence_array = np.array(list(coherence_array.values()))

#%%
#relative amplitude for a single model depending on the target cell population

model_nb = 36
hem_stim = 'hem1'
hem_record = 'hem1'
labels = list(coherences['hem1']['none_00'].keys())

for i in range(4):
    plt.plot(coherence_array[Hems[hem_stim],:,i,model_nb,Hems[hem_record]], marker='o', label=labels[i])

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['n', 'i25', 'i50', 'i75', 'i100', 'c25', 'c50', 'c75', 'c100', 'b25', 'b50', 'b75', 'b100'])
plt.title(f'{file_order[model_nb]}')
plt.legend()

#%%
#mean for all models with stim and record hemisphere

hem_stim = 'hem1'
hem_record = 'hem1'

labels = list(coherences['hem1']['none_00'].keys())
data=np.mean(coherences_array_75[Hems[hem_stim],:,:,:,Hems[hem_record]], axis=2)
for i in range(4):
    plt.plot(data[:,i], marker='o', label=labels[i])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['n', 'i25', 'i50', 'i75', 'i100', 'c25', 'c50', 'c75', 'c100', 'b25', 'b50', 'b75', 'b100'])
plt.title(f'mean for sparse model')
plt.legend()


#%% compare distribution of variance and mean response between model_info and model_best file

mean_resp_all = np.array(np.mean(model_info[list(columns[9:13])], axis=1))
mean_resp_best = np.array(np.mean(model_best[list(columns[9:13])], axis=1))

resp_all_dens = gaussian_kde(list(mean_resp_all))
resp_best_dens = gaussian_kde(list(mean_resp_best))

#xs = np.linspace(-0.3, 0.02, 100)
xs = np.linspace(0.1, 0.5, 100)

dens, ax = plt.subplots(1,1)
ax.set_title('mean response distribution')
ax.hist([mean_resp_all, mean_resp_best],  bins=10 , color=['red', 'blue'], label = ['all models', 'best models'])
#ax.plot(xs, resp_all_dens(xs), c='red', label='all')
#ax.plot(xs, resp_best_dens(xs), c='blue', label='best')
ax.legend()
plt.show()

#%%

total_hem1 = np.array(third_set['nb_hem1_ipsi_pref'] + third_set['nb_hem1_contra_pref'])
norm = colors.Normalize(vmin=0, vmax=total_hem1.max())

fig_ratio,ax = plt.subplots(1,1,figsize=(6,6))

for i in list(coherences.keys()):
    if i[14] == '7' or i[14]=='5':
        ratio = coherences[i][:,0]/(coherences[i][:,1]-coherences[i][:,0])
        percent = (coherences[i][:,0]/coherences[i][:,1])*100
        percent[np.isnan(percent)] = 0
        ax.scatter(ratio, coherences[i][:,2]-coherences[i][:,3], s=20, alpha=0.8)

ax.set_xlabel('ratio ipsi/contra')
divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig_ratio.add_axes(ax_cb)
cb1 = colorbar.ColorbarBase(ax_cb, norm=norm, orientation='vertical', label='total activated cells')
#ax.set_xlim(-0.04,2.04)
ax.set_ylabel('<- choice 2 - choice 1 ->')
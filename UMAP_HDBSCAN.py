# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:28:50 2023

@author: Bear
"""

"""1) Apro Dataframe"""
import pandas as pd

df=pd.read_csv(r'C:\Users\usr02709\Desktop\Materiale\Materiale TMA\UMAP\TMA112BB1A_30 markers detection.csv',sep=';')
colonne=df.columns

d=df[df['TMA core']=='D-5']
a=d.iloc[:,0:4]
a['FAP1 non sub']=d['FAP1: Cell: Mean']
a['AF']=d['AF (R4): Cell: Mean']

a['FAP1']=d['FAP1: Cell: Mean']-d['AF (R4): Cell: Mean']

"""2) Creo il dataframe che mi serve per fare la UMAP."""
#Creo DataFrame con marcatori sottratti dall AF.

#Creo df in cui inserire i marcatori sottratti
sub_df=df.iloc[:,0:4] #Qui ho se la cellula è normale o fibrotica, l'identificativo del core e le coordinate.


#Aggiugo i marcatori sottratti
#Round2
#sub_df['LPAR1']=df['LPAR1: Cell: Mean']-df['AF (R2): Cell: Mean']
#Round3
sub_df['PU1 Nucleus']=df['PU1: Nucleus: Mean']-df['AF (R3): Nucleus: Mean']
#sub_df['LPA2']=df['LPA2: Cell: Mean']-df['AF (R3): Cell: Mean']
#Round4
#sub_df['FAP1']=df['FAP1: Cell: Mean']-df['AF (R4): Cell: Mean']
sub_df['Coll1']=df['Coll1: Cell: Mean']-df['AF (R4): Cell: Mean']
sub_df['PanCK']=df['PanCK: Cell: Mean']
#Round5
sub_df['HOP']=df['HOP: Cell: Mean']-df['AF (R5): Cell: Mean']
sub_df['aSMA']=df['aSMA: Cell: Mean']-df['AF (R5): Cell: Mean']
sub_df['PAX5 Nucleus']=df['PAX5: Nucleus: Mean']
#Round6
sub_df['CD3']=df['CD3: Cell: Mean']-df['AF (R6): Cell: Mean']
#Round7
sub_df['CD206']=df['CD206: Cell: Mean']-df['AF (R7): Cell: Mean']
sub_df['CD68']=df['CD68: Cell: Mean']
#Round8
sub_df['Byglican']=df['Byglican: Cell: Mean']
#Round9
sub_df['TTF1 Nucleus']=df['TTF1: Nucleus: Mean']-df['AF (R9): Nucleus: Mean']
sub_df['ABCA3']=df['ABCA3: Cell: Mean']-df['AF (R9): Cell: Mean']
sub_df['CD4']=df['CD4: Cell: Mean']
#Round10
#sub_df['Caspase3']=df['Caspase3: Cell: Mean']-df['AF (R10): Cell: Mean']
#Round11
#sub_df['CXCL10']=df['CXCL10: Cell: Mean']-df['AF (R11): Cell: Mean']
sub_df['CCR2']=df['CCR2: Cell: Mean']
#Round12
sub_df['PDPN']=df['PDPN: Cell: Mean']-df['AF (R12): Cell: Mean']
sub_df['Tubulin']=df['Tubulin: Cell: Mean']-df['AF (R12): Cell: Mean']
sub_df['Coronin1a']=df['Coronin1a: Cell: Mean']
#Round13
sub_df['AQP5']=df['AQP5: Cell: Mean']-df['AF (R13): Cell: Mean']
sub_df['EpCAM']=df['EpCAM: Cell: Mean']-df['AF (R13): Cell: Mean']
sub_df['VWF']=df['VWF: Cell: Mean']
sub_df['AQP5 Nucleus']=df['AQP5: Nucleus: Mean']-df['AF (R13): Nucleus: Mean']
sub_df['EpCAM Nucleus']=df['EpCAM: Nucleus: Mean']-df['AF (R13): Nucleus: Mean']
#Round14
sub_df['FOXJ1']=df['FOXJ1: Cell: Mean']-df['AF (R14): Cell: Mean']
sub_df['SCGB1A1']=df['SCGB1A1: Cell: Mean']-df['AF (R14): Cell: Mean']
sub_df['OPN']=df['OPN: Cell: Mean']
sub_df['FOXJ1 Nucleus']=df['FOXJ1: Nucleus: Mean']-df['AF (R14): Nucleus: Mean']
#Round15
sub_df['Fibronectin']=df['Fibronectin: Cell: Mean']-df['AF (R15): Cell: Mean']
sub_df['FOXP3']=df['FOXP3: Cell: Mean']
sub_df['FOXP3 Nucleus']=df['FOXP3: Nucleus: Mean']


#Scelgo i campioni
sample=sub_df[(sub_df['TMA core']=='B-4') & (sub_df['TMA core']=='B-4')]

#Se volessi mettere assieme più core fai così.
#sample=sub_df[(sub_df['TMA core']=='B-4') & (sub_df['TMA core']=='Altro codice di un altro core') & ecc.]

#Creo DataFrame per quando dovrò plottare marcatori su UMAP. 

Marcatori=df.iloc[:,0:4]

#Aggiugo i marcatori sottratti
#Round2
#sub_df['LPAR1']=df['LPAR1: Cell: Mean']-df['AF (R2): Cell: Mean']
#Round3
Marcatori['PU1 Nucleus']=df['PU1: Nucleus: Mean']-df['AF (R3): Nucleus: Mean']
#Marcatori['LPA2']=df['LPA2: Cell: Mean']-df['AF (R3): Cell: Mean']
#Round4
Marcatori['FAP1']=df['FAP1: Cell: Mean']-df['AF (R4): Cell: Mean']
Marcatori['Coll1']=df['Coll1: Cell: Mean']-df['AF (R4): Cell: Mean']
Marcatori['PanCK']=df['PanCK: Cell: Mean']
#Round5
Marcatori['HOP']=df['HOP: Cell: Mean']-df['AF (R5): Cell: Mean']
Marcatori['aSMA']=df['aSMA: Cell: Mean']-df['AF (R5): Cell: Mean']
Marcatori['PAX5 Nucleus']=df['PAX5: Nucleus: Mean']
#Round6
Marcatori['CD3']=df['CD3: Cell: Mean']-df['AF (R6): Cell: Mean']
#Round7
Marcatori['CD206']=df['CD206: Cell: Mean']-df['AF (R7): Cell: Mean']
Marcatori['CD68']=df['CD68: Cell: Mean']
#Round8
Marcatori['Byglican']=df['Byglican: Cell: Mean']
#Round9
Marcatori['TTF1 Nucleus']=df['TTF1: Nucleus: Mean']-df['AF (R9): Nucleus: Mean']
Marcatori['ABCA3']=df['ABCA3: Cell: Mean']-df['AF (R9): Cell: Mean']
Marcatori['CD4']=df['CD4: Cell: Mean']
#Round10
Marcatori['Caspase3']=df['Caspase3: Cell: Mean']-df['AF (R10): Cell: Mean']
#Round11
Marcatori['CXCL10']=df['CXCL10: Cell: Mean']-df['AF (R11): Cell: Mean']
Marcatori['CCR2']=df['CCR2: Cell: Mean']
#Round12
Marcatori['PDPN']=df['PDPN: Cell: Mean']-df['AF (R12): Cell: Mean']
Marcatori['Tubulin']=df['Tubulin: Cell: Mean']-df['AF (R12): Cell: Mean']
Marcatori['Coronin1a']=df['Coronin1a: Cell: Mean']
#Round13
Marcatori['AQP5']=df['AQP5: Cell: Mean']-df['AF (R13): Cell: Mean']
Marcatori['EpCAM']=df['EpCAM: Cell: Mean']-df['AF (R13): Cell: Mean']
Marcatori['VWF']=df['VWF: Cell: Mean']
Marcatori['AQP5 Nucleus']=df['AQP5: Nucleus: Mean']-df['AF (R13): Nucleus: Mean']
Marcatori['EpCAM Nucleus']=df['EpCAM: Nucleus: Mean']-df['AF (R13): Nucleus: Mean']
#Round14
Marcatori['FOXJ1']=df['FOXJ1: Cell: Mean']-df['AF (R14): Cell: Mean']
Marcatori['SCGB1A1']=df['SCGB1A1: Cell: Mean']-df['AF (R14): Cell: Mean']
Marcatori['OPN']=df['OPN: Cell: Mean']
Marcatori['FOXJ1 Nucleus']=df['FOXJ1: Nucleus: Mean']-df['AF (R14): Nucleus: Mean']
#Round15

Marcatori['Fibronectin']=df['Fibronectin: Cell: Mean']-df['AF (R15): Cell: Mean']
Marcatori['FOXP3']=df['FOXP3: Cell: Mean']
Marcatori['FOXP3 Nucleus']=df['FOXP3: Nucleus: Mean']

Marcatori=Marcatori[Marcatori['TMA core']=='B-4']



#Trasformo i valori negativi in 0 per il dataframe che uso per fare la UMAP
valori=sample.iloc[:,4:]
valori=valori.where(valori>0,0)


#Trasformo i valori negativi in 0 per il dataframe che uso per plottare sulla UMAP i marcatori che voglio
Marcatori=Marcatori.iloc[:,4:]
Marcatori=Marcatori.where(valori>0,0)


"""3)Normalizzo i Dati e creo UMAP"""
#Normalizzo i dati e li clasterizzo con UMAP
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


#1)Normalizzo i dati con normalizzazione single cell
#Median Normalization
num_transcripts = np.sum(valori, axis=0)
X_norm = (np.median(num_transcripts) / num_transcripts) * valori

#Freeman_tukey_trasformation
df_norm=np.sqrt(X_norm) + np.sqrt(X_norm+1)

#2)Normalizzo i dati con Z-Score
df_norm = StandardScaler().fit_transform(valori)

#Faccio PCA
pca_scores=PCA().fit_transform(df_norm)
df_pc=pd.DataFrame(pca_scores)


#Faccio UMAP
#dist=[0.0,0.1,0.25,0.5,0.8,0.99]
#for min_d in dist:

min_d=0.0
reducer = umap.UMAP(n_neighbors=50, metric='euclidean',min_dist=min_d,random_state=42)
embedding = reducer.fit_transform(df_pc)

embedding.shape

"""4)Faccio i primi grafici con UMAP.
Plotto la UMAP poi la plotto con sopra l'informazione se una cellula è fibrotica o normale."""
#Creato dataframe con coordinate UMAP.
embedding=pd.DataFrame(embedding,columns=['x','y'])


#Plottiamo UMAP
plt.scatter(
    embedding.iloc[:, 0],
    embedding.iloc[:, 1],s=0.1)
#c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP, min_dist = {}'.format(min_d), fontsize=24)
plt.show()




#Plotto UMAP connormale fibrotico
#Creo lista coi colori per normale e fibrotico
fibrotic_color=[]
for i in sample['Class']:
    if i=='Normal':
        fibrotic_color.append('green')
    if i=='Fibrotic':
        fibrotic_color.append('brown')

#Plotto il grafico
plt.scatter(
    embedding.iloc[:, 0],
    embedding.iloc[:, 1],c=fibrotic_color,s=2)   
plt.scatter([],[],label='Normal',color='green')
plt.scatter([],[],label='Fibrotic',color='brown')
plt.legend(bbox_to_anchor=(1.02, 1))
plt.show()


"""5)Classifico le mie cellule nei cluster con HDBSCAN."""
#Facciamo classificazione con HDBSCAN
import time
import hdbscan
from random import randint

#Ricreo dataframe con coordinate UMAP, così posso rilanciare HDBSCAN dopo aver plottato la UMAP coi Cluster.
#Se non lo rifacessi nel dataframe embedding che contiene le coordinate UMAP avrei la colonna dei cluster e dei colori.
#E HDBSCAN non andrebbe, se lo fai prima di lanciare HDBSCAN la prima volta, non c'è problema, va comunque.
embedding=pd.DataFrame(embedding,columns=['x','y'])
#Lancio HDBSCAN
labels = hdbscan.HDBSCAN(20).fit_predict(embedding) #Ho fatto la classificazione dei cluster con hdbscan


"""6)Plotto la UMAP coi Colori dei cluster.
Prima la plotto con anche il cluster rumore -1 e poi senza."""
#Creo Colori per i cluster
colors=[]
for i in range(len(labels)):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

#Inserisco nel DF
embedding['clusters']=labels

embedding['colors']=embedding['clusters'].map(lambda x: colors[x])


#Faccio grafico con Matplotlib CONSIGLIATO

#Creo liste per i colori e i labels così da creare la legenda.
#Creo anche lista con coordinate e colori per ogni cellulina da inserire nello scatter.
colors=embedding['colors'].unique()
clusterini=embedding['clusters'].unique()
x,y,color=embedding['x'],embedding['y'],embedding['colors']
#Creo legenda
markers=[]
for colori,label in zip(colors,clusterini):
    markers.append(plt.scatter([],[], c=colori, label=label))


#Faccio il grafico
plt.scatter(x,y,s=2,c=color)
plt.legend(handles=markers, bbox_to_anchor=(1, 1))
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()



#Altro metodo per il grafico in cui plotto UMAP coi cluster, Uso Seaborn.COME INDICATO SOPRA MEGLIO USARE MATPLOTLIB
#Faccio grafico con Seaborn
fig = plt.figure(dpi=140)
sns.set()
sns.lmplot(x='x',y='y',data=embedding,hue='clusters',fit_reg=False,legend=True,height=5)
plt.show()



#Escludo rumore dai grafici UMAP e poi riplotto 
embedding['Class_color']=fibrotic_color
no_noise=embedding[embedding['clusters']!=-1]


#Faccio grafico con Matplotlib CONSIGLIATO

#Creo liste per i colori e i labels così da creare la legenda.
#Creo anche lista con coordinate e colori per ogni cellulina da inserire nello scatter.
colors=no_noise['colors'].unique()
clusterini=no_noise['clusters'].unique()
x,y,color=no_noise['x'],no_noise['y'],no_noise['colors']
#Creo legenda
markers=[]
for colori,label in zip(colors,clusterini):
    markers.append(plt.scatter([],[], c=colori, label=label))

    
#Faccio il grafico senza rumpre
plt.scatter(x,y,s=0.1,c=color)
plt.legend(handles=markers, bbox_to_anchor=(1, 1))
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()


"""7)Plotto le cellule nei vari cluster su tessuto TMA."""
#plotta i clusters su TMA
sample['cluters']=list(labels)
for i in clusterini:    
    #i=12   
    dfsc=sample[sample['cluters']==i]    
    plt.scatter(sample['Centroid X µm'],sample['Centroid Y µm'],color='b',s=2)
    plt.scatter(dfsc['Centroid X µm'],dfsc['Centroid Y µm'], color='r',s=2)
    plt.title('cluster numero {}'.format(i))
    plt.gca().invert_yaxis()
    plt.show()

"""8)Plotto i marcatori su UMAP"""
#Trovo per i vari marcatori il valore massimo e minimo.
print(Marcatori['Coll1'].min(),Marcatori['Coll1'].max())

#Faccio per i marcatoeri che voglio plottare su UMAP l'histogramma di intensità.
#Così posso trovare la soglia anche in modo visivo
plt.hist(Marcatori['Coll1'])
plt.title('Coll1')
plt.plot()


#Prendo coordinate di UMAP
x,y=embedding.iloc[:,0],embedding.iloc[:,1]

#Vmin e Vmax sono le soglie dell'intensità.
plt.scatter(x,y, c=Marcatori['Coll1'], cmap='cividis',s=0.1,vmin=0,vmax=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar()
#plt.xlim([-40,40])
plt.title('Coll1')
plt.show()  



"""9)Faccio le clustermap"""
#Faccio Clustermap
column=valori.columns
dfs=pd.DataFrame(df_norm,columns=column)
dfs['clusters']=labels
dfs['Class']=list(sample['Class'])
dfs['Class_color']=fibrotic_color
dfs=dfs[dfs['clusters']!=-1]
cluster_unici=list(dfs['clusters'].unique())

for cluster in cluster_unici:
    sottogruppo=dfs[dfs['clusters']==cluster]
    fibrotic_target=sottogruppo['Class_color']
    sottogruppo=sottogruppo.iloc[:,:29]
    sottogruppo=sottogruppo.T

    sns.set(rc={'figure.figsize':(15,30)})
    h=sns.clustermap(sottogruppo,xticklabels=False,yticklabels=True,cmap="vlag",cbar_kws={'label':f'{cluster}'},col_colors=fibrotic_target,vmin=-1,vmax=1)
    plt.scatter([],[],label='Normal',color='green')
    plt.scatter([],[],label='Fibrotic',color='brown')
    plt.legend(bbox_to_anchor=(20, 1), title="Class_color")
    plt.show()




"""10)Plotto la UMAP con i cluster rinominati col tipo cellulare."""

label=pd.read_excel(r'C:\Users\usr02709\Desktop\Materiale\Materiale TMA\csv per Davide\TMA112BB11A\Risultati Python Full ROI\label\BLM 21D (tutte)_label.xlsx')

l=label['Cell Type'].unique()
dictionary=dict(zip(label['Cluster'], label['Cell Type']))
no_noise['Cell type']= no_noise['clusters'].map(lambda x: dictionary[x])

#plotto il t-sne
cell_colors=['#ef83bd','#7f4d82','#cd3333','#82b280','#99b3fe','#959595','#ff9900','#76ff00','#8bfbff','#ffa58b','#006c8b','#000000','#FF1493','#ADFF2F','#FFD700','#71C671','#8B1A1A']
cell_type=['B Cells','Macrophages','Muscle Cells','Stromal Cells','Epithelial Cells','Unknown','T Cells','Th Cells','Monocytes Macrophages','Monocytes','Fibroblast','Inflammatory Cells','Leukocytes','Myofibroblast','Th Cells Naive','Th Cells Primed','PB Fibroblast']


dictionary_cc=dict(zip(cell_type, cell_colors))
no_noise['Cell color']= no_noise['Cell type'].map(lambda x: dictionary_cc[x])  


# creo la lista dei label per la legenda del grafico
x,y,color=no_noise.iloc[:,0],no_noise.iloc[:,1],no_noise.iloc[:,5]
cell_type,cell_color=no_noise.iloc[:,4].unique(),no_noise.iloc[:,5].unique()
markers = []
for colori,label in zip(cell_color,cell_type):
    markers.append(plt.scatter([],[], c=colori, label=label))
plt.scatter(x,y,s=2,c=color)
plt.legend(handles=markers, bbox_to_anchor=(1, 1))
#plt.legend()
plt.show()




"""11)Calcolo le cellule positive ad un dato marcatore all'interno di un cluster, con e senza treshold."""

#Faccio le percentuali di positività al fap non sottratto dell'autofluo per ogni gruppo cellulare
#E creo il dataframe
Marcatori['clusters']=labels
Marcatori = Marcatori[Marcatori['clusters'] != -1]
r=list(Marcatori['FAP1'])

no_noise['Fap1 non sott']=r
nomi_cellule=list(no_noise['Cell type'].unique())
dataframe=pd.DataFrame(columns=nomi_cellule)

for i in range(len(nomi_cellule)):
    marcatore=no_noise[no_noise['Cell type']==nomi_cellule[i]]
    fap=marcatore[marcatore['Fap1 non sott']>=20]
    percentuale=(len(fap)/len(marcatore))*100
    
    dataframe.loc[0,f'{nomi_cellule[i]}'] = percentuale
dataframe.to_excel(r'C:\Users\usr02709\Desktop\Materiale\Materiale TMA\csv per Davide\TMA112BB11A\Risultati Python Full ROI\Nint E1 E2 E3 E4 E7 E8\Percentuale positività fap nei cluster.xlsx', index=False)

marcatore=no_noise[no_noise['Cell type']=='Leukocytes']


#Cerchiamo le Fap pos totali con treshold.
percframe=pd.DataFrame()
Fap1=no_noise[no_noise['Fap1']>=4.423]
for i in nomi_cellule:
    tipo_cell=Fap1[Fap1['Cell type']==i]
    perc=(len(tipo_cell)/len(Fap1))*100
    percframe.loc[0,f'{i}'] = perc
percframe.to_excel(r'C:\Users\usr02709\Desktop\Materiale\Materiale TMA\csv per Davide\TMA112BB11A\Risultati Python Full ROI\Bleo 28D B8 C7 C8 D7 D8\Percentuale cellule Fap per tipo cellulare sul totale cellule Fap per Bleo 28D.xlsx', index=False)


#!/usr/bin/env python
# coding: utf-8

# In[11]:


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt


# # Matplotlib Backends - Inline

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


plt.plot(5, 5, 'o')
plt.show()


# In[118]:


import pandas as pd
import matplotlib as mpl
from __future__ import print_function #adds compatibility to python 2


# In[119]:


#install xlrd


print('xlrd installed!')


# In[208]:


df_can = pd.read_excel('Canada.xlsx',"Canada by Citizenship", skiprows = range(20), skip_footer = 2)


# In[209]:


df_can.head()


# In[210]:


df_can.tail()


# In[211]:


#to get basic information about the dataset
df_can.info()


# In[212]:


#to get the list of all columns existing in the dataset
df_can.columns.values


# In[213]:


df_can.index.values


# In[214]:


print(type(df_can.columns))
print(type(df_can.index))


# In[215]:


#as the default type of index and columns are not a list we use the tolist() method to transform them
df_can.columns.tolist()
df_can.index.tolist()

print(type(df_can.columns))
print(type(df_can.index))


# In[216]:


#lets see the dimension of the dataframe
df_can.shape


# # Cleaning the Dataset

# In[218]:


#in pandas axis = 0 are rows and axis = 1 are columns
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace = True)
df_can.head(2)


# In[130]:


#rename columns
df_can.rename(columns={'OdName':'Country', 'AreaName': 'Continent', 'RegName': 'Region'}, inplace = True)
df_can.columns


# In[131]:


#add a Total Column to sum up the total of immigrants by country over the entire period 1980 to 2013
df_can['Total'] = df_can.sum(axis = 1)


# In[132]:


df_can.head()


# In[133]:


#checking how many null objects we have
df_can.isnull().sum()


# In[134]:


#quick summary of each column
df_can.describe()


# In[135]:


df_can.Country


# In[136]:


#list of countries from 1980 to 1985
df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]]


# In[137]:


#setting 'Country' as the index 
df_can.set_index('Country', inplace = True)

#the oposite of set_index is reset_index


# In[138]:


df_can.head(3)


# In[139]:


# if we want to remove the name of the index
#df_can.index.name = None


# view the number of immigrants from Japan for the following scenarios:
# 
# 1. The full row data (all columns)
# 2. For year 2013
# 3. For years 1980 to 1985

# In[140]:


#1 the full row data 
print(df_can.loc['Japan'])

# other ways
#print(df_can.iloc[87]) #if we know the position
#print(df_can[df_can.index == 'Japan'].T.squeeze())


# In[141]:


#2 for year 2013
print(df_can.loc['Japan', 2013])


# In[142]:


#3 for years 1980 to 1985
print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1985]])


# In[143]:


#columns names that are integers can be confusing the best practice is to name them as strings
df_can.columns= list(map(str, df_can.columns))


# In[144]:


#to check type of columns headers
[print (type(x)) for x in df_can.columns.values]


# In[168]:


#as now the column years are string we will 
#declare a variable that will allow us to easily call upon the full range of years:
years = list(map(str, range(1980, 2014)))
#years


# # Filtering based on a Criteria

# In[146]:


#filtering the dataframe to show the data on Asian countries (AreaName = Asia)
#1 create the condition boolean series
condition = df_can['Continent'] == 'Asia'
print(condition)


# In[147]:


#2 pass this condition into the dataframe
df_can[condition]


# In[148]:


df_can[(df_can['Continent'] == 'Asia') & (df_can['Region'] == 'Southern Asia')]


# In[149]:


#review the changes we have made to the dataframe 
print('data dimensions: ', df_can.shape)
print(df_can.columns)
df_can.head(2)


# # Visualizing Data using Matplotlib

# In[150]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt

#apply style to Matplotlib
print(plt.style.available)
mpl.style.use(['ggplot'])


#  Plot a line graph of immigration from Haiti

# In[151]:


haiti = df_can.loc['Haiti', years]
haiti.head()


# In[152]:


haiti.plot()


# In[153]:


haiti.index= haiti.index.map(int) # changing the index values to type integer for plotting
haiti.plot(kind='line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()#we need this line to show the updates done


# In[154]:


haiti.plot(kind = 'line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

##annotate the 2010 Earthquake
#syntax: plt.text(x, y, label)
plt.text(2000, 6000, '2010 Earthquake')

plt.show()


# Compare the number of immigrants from India and China from 1980 to 2013.

# In[155]:


df_CI = df_can.loc[['India', 'China'], years]
df_CI.head()


# In[165]:


#df_CI.plot(kind='line')


# In[157]:


#it does'nt look right, lets use transpose to swap the rows and columns
df_CI = df_CI.transpose()
df_CI.head()


# In[158]:


haiti.index= haiti.index.map(int) # changing the index values to type integer for plotting
df_CI.plot(kind='line')

plt.title('Immigration from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# Compare the trend of top 5 countries that contributed the most to immigration to Canada.

# In[178]:


# Step 1: Get the dataset. Recall that we created a Total column that 
#calculates the cumulative immigration by country.
#We will sort on this column to get our top 5 countries using pandas sort_values() method.
#inplace = True paramemter saves the changes to the original df_can dataframe
df_can.sort_values(['Total'], ascending = False, axis = 0, inplace = True)


# In[192]:



#define the top 5 countries
df_top5 = df_can.head()


# In[193]:



## transpose the dataframe
df_top5 = df_top5[years].transpose()


# In[188]:



df_top5


# In[ ]:





# In[ ]:





# In[ ]:





# In[179]:


# Step 2: Plot the dataframe. To make the plot more readeable, 
#we will change the size using the `figsize` parameter.
df_top5.index = df_top5.index.map(int) # let's change the index values of df_top5 to type integer for plotting
df_top5.plot(kind='area', figsize =(14, 8))# pass a tuple (x, y) size

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# In[180]:


#!conda install -c anaconda xlrd --yes


# In[182]:


df_canada = pd.read_excel('Canada.xlsx', 'Canada by Citizenship', skiprows = range(20), skipfooter = 2)
print('Data dawloaded and read into a dataframe')


# In[183]:


df_canada.head()


# In[184]:


#print dimensions of the dataframe
print(df_canada.shape)


# Clean up the dataset to remove columns that are not informative to us for visualization 

# In[185]:


df_canada.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1, inplace=True)
df_canada.head()


# In[219]:


#Rename the columns
df_canada.rename(columns={'OdName': 'Country','AreaName': 'Continent','RegName': 'Region'}, inplace=True)


# In[220]:


#Making sure all column labels are type string
#1 examine the type of them
all(isinstance(column, str) for column in df_canada.columns)


# In[221]:


#changing them all into string
df_canada.columns = list(map(str, df_canada.columns))

#checking the columns label types
all(isinstance(column, str) for column in df_canada.columns)


# In[222]:


#Set the country name as index
df_canada.set_index('Country', inplace=True)


# In[223]:


#add a Total Columns
df_canada['Total']=df_canada.sum(axis=1)


# In[224]:


print('Data Dimensions', df_canada.shape)


# In[225]:


#Create a list of years from 1980-2013
years = list(map(str, range(1980, 2014)))

years


# In[226]:


#Adding style
mpl.style.use('ggplot') 


# In[227]:


df_canada.sort_values(['Total'], ascending = False, axis = 0, inplace=True)

#get the top 5 countries
df_top5_new = df_canada.head()


# In[228]:


df_top5_new.head()


# In[229]:


#transpose the dataframe
df_top5_new = df_top5_new[years].transpose()

df_top5_new.head()


# In[230]:


# Lets change the type of index for integer for plottin
df_top5_new.index = df_top5_new.index.map(int)

df_top5_new.plot(kind='area', stacked = False, figsize =(20,10)#pass a tuple (x,y) size
                )

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# In[231]:


# preferred option for plotting with more flexibility
ax = df_top5_new.plot(kind = 'area', alpha = 0.35, figsize=(20,10))

ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')


# # 5 countries that contributed the least to immigration to Canada from 1980 to 2013

# In[233]:


df_canada.sort_values(['Total'], ascending = False, axis = 0, inplace=True)

#get the top 5 countries
df_top5_least = df_canada.tail()


# In[234]:


df_top5_least


# In[235]:


#transpose the dataframe
df_top5_least = df_top5_least[years].transpose()

df_top5_least.head()


# In[238]:


# Lets change the type of index for integer for plottin
df_top5_least.index = df_top5_least.index.map(int)

df_top5_least.plot(kind='area',alpha= 0.45, figsize =(20,10)#pass a tuple (x,y) size
                )

plt.title('Immigration Trend of 5 Countries with Least Contribution to Immigration')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()


# Use the artist layer to create an unstacked area plot of the 5 countries that contributed the least to immigration to Canada

# In[239]:


ax = df_top5_least.plot(kind = 'area', alpha= 0.55, figsize=(20,10))

ax.set_title('Immigration Trend of 5 Countries with Least Contribution to Immigration')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')


# # HISTOGRAMS

# # What is the frequency distribution of the number (population) of new immigrants from the various countries to Canada in 2013?

# In[241]:


#view the 2013 data
df_canada['2013'].head(10)


# In[242]:


#np.hitogram returns 2 values
count, bin_edges = np.histogram(df_canada['2013'])

print(count)#frequency count
print(bin_edges)#bin ranges default  = 10 bins


# In[243]:


#easily graph this distribution by passing kind=hist to plot()
df_canada['2013'].plot(kind='hist', figsize=(8,5))

plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()


# Adjusting the bin sizes

# In[245]:


count, bin_edges = np.histogram(df_canada['2013'])

df_canada['2013'].plot(kind='hist', figsize=(8,5), xticks=bin_edges)

plt.title('Histogram of Immigration from 195 countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

plt.show()


# # What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013

# In[246]:


#quicly view of the dataset
df_canada.loc[['Denmark','Norway','Sweden'], years]


# In[247]:


df_canada.loc[['Denmark','Norway','Sweden'], years].plot.hist()


# In[250]:


#lets transpose the dataset first then we plot
df_t = df_canada.loc[['Denmark','Norway','Sweden'], years].transpose()
df_t.head()


# In[251]:


df_t.plot(kind='hist', figsize=(10,6))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()


# increase the bin size to 15 by passing in bins parameter<br>
# set transparency to 60% by passing in alpha paramemter<br>
# label the x-axis by passing in x-label paramater<br>
# change the colors of the plots by passing in color paramete<br>

# In[256]:


#lets get the x-tick values
count, bin_edges = np.histogram(df_t, 15)

#un-stacked histograms
df_t.plot(kind='hist',
          figsize=(10, 6), 
          bins=15,
          alpha=0.6,
          xticks=bin_edges, 
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()


# #  we do no want the plots to overlap each other

# In[257]:


count, bin_edges = np.histogram(df_t, 15)

xmin = bin_edges[0] - 10 #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes
xmax = bin_edges[-1] + 10 #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

#stacked histogram
df_t.plot(kind='hist',
         figsize=(10,6),
         bins=15,
         xticks=bin_edges,
         color=['coral', 'darkslateblue', 'mediumseagreen'],
         stacked=True,
         xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()


# # Use the scripting layer to display the immigration distribution for Greece, Albania, and Bulgaria for years 1980 - 2013

# In[258]:


#quicly view of the dataset
df_canada.loc[['Greece','Albania','Bulgaria'], years]


# In[260]:


#lets transpose the dataset first then we plot
df_t2 = df_canada.loc[['Greece','Albania','Bulgaria'], years].transpose()
df_t2.head()


# In[261]:


count, bin_edges = np.histogram(df_t2, 15)

xmin = bin_edges[0] - 10 #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes
xmax = bin_edges[-1] + 10 #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

#stacked histogram
df_t.plot(kind='hist',
         figsize=(10,6),
         bins=15,
         xticks=bin_edges,
         color=['coral', 'darkslateblue', 'mediumseagreen'],
         stacked=True,
         xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()


# # Bar Charts(Dataframe)

# Let's compare the number of Iceland immigrants (country = 'Iceland') to Canada from year 1980 to 2013

# In[262]:


#step1 get the data
df_iceland = df_canada.loc['Iceland', years]
df_iceland.head()


# In[263]:


#step2 plot data
df_iceland.plot(kind='bar', figsize=(10,6))

plt.title('Icelandic immigrants to Canada from 1980 to 2013')
plt.ylabel('Number of immigrants')
plt.xlabel('Year')

plt.show()


# The bar plot above shows the total number of immigrants broken down by each year. We can clearly see the impact of the financial crisis; the number of immigrants to Canada started increasing rapidly after 2008.

# In[266]:


df_iceland.plot(kind='bar', figsize=(10,6), rot=90)#rotate the bars by 90 degrees

plt.xlabel('Years')
plt.ylabel('Number of immigrants')
plt.title('Icelandic immigrants to Canada from 1980 to 2013')

#annotate arrow
plt.annotate('',# s: str. Will leave it blank for no text
            xy=(32, 70), # place head of the arrow at point (year 2012 , pop 70)
            xytext=(28, 20), # place base of the arrow at point (year 2008 , pop 20)
            xycoords='data',# will use the coordinate system of the object being annotated
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

#annotate text
plt.annotate('2008 - 2011 Financial Crisis',# text to display
              xy=(28, 30), # start the text at at point (year 2008 , pop 30)
              rotation=72.5,# based on trial and error to match the arrow
              va='bottom',# want the text to be vertically 'bottom' aligned
              ha='left')# want the text to be horizontally 'left' algned.

plt.show()


# # Using the scripting layter and the df_can dataset, create a horizontal bar plot showing the total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013. Label each country with the total immigrant count.

# In[273]:


df_canada.sort_values(['Total'], ascending = True, inplace=True)

#get the top 15 countries
df_top15 = df_canada['Total'].tail(15)

#transpose the dataframe
#df_top15 = df_top15[years].transpose()

df_top15


# In[277]:


#step2 plot data
df_top15.plot(kind='barh', figsize=(12,12), color='steelblue')

plt.title('Top 15 countries of immigrants to Canada from 1980 to 2013')
plt.xlabel('Number of immigrants')

# annotate value labels to each country
for index, value in enumerate(df_top15):
    label=format(int(value), ',')  # format int with commas
    
    # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(4700, index - 0.10), color='white')
    

plt.show()


# In[ ]:





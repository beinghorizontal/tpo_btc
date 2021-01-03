# tpo_btc
This is a Python visualization code. It fetches BTC data from Binance servers and streams live market profile chart using Dash. This code is the modification of my existing old code posted here https://github.com/beinghorizontal/tpo_project/

## Dependencies
pip install plotly (update if previously installed)

pip install dash (update if previously installed)

pip install pandas

pip install requests

## Steps for live chat

Download all the files or clone this repo as a zip and extract it in a local folder.

In Windows: a) Edit btc_mp_v2.bat file (right-click, open in notepad) and change the URL where you saved btc_mp_v2.py and save it
eg. start "" http://127.0.0.1:8000/ & python.exe C:\here_i_saved_the_file\btc_mp_v2.py
b) Double click the .bat file and a new tab will open in your default browser, wait for few seconds as it takes time to fetch the data from the servers and finish the one-time calculations for the context (this delay is only when you first run the batch script) and live chat will get automatically updated. That's it.

In Linux: a) Edit btc_mp_v2.sh  and change the URL where you saved btc_mp_v2.py and save it.
b) You have to make this shell script executable. Open the command prompt, cd to where you saved the python file and type chmod +x btc_mp_v2.py
c) Double click the .sh file and a new tab will open in your default browser, wait for few seconds as it takes time to fetch the data from the servers and finish the one-time calculations for the context (this delay is only when you first run the shell script) and live chat will get automatically updated. That's it.

By default only the current day is displayed, if you want to see more than one day then adjust the slider at the bottom (see the 1st image below)

If you want to use volume profile instead of TPO profile (based on price), then edit the code btc_mp_v1.py and go to the line 55 and replace mode ='tpo' by mode ='vol'. You don't need to close the live chart if it is open, chart will automatically refresh (Unless you executed the code from ipython console) and the code will use volume for all market profile related calculations. 

## What is new

 The old code was for the local data while this one fetches the live data for BTC-USD directly from Binance servers in 30 minute format for last 10 days (binance limitation) You can use it for any other pair supported by Binance. Fetching the data part is actually very small. Any URL that supports 30-minute data with a minimum history of 10 days will work with this code.

Wrote one big class for all market profile and day ranking related calculations instead of functions which is the heart of the code. It also means no repeat calculations and more readable code

Added volume bars at the bottom

Removed Initial Balance (Opening range) related calculations as it was impacting the stability of the code and also it doesn't make sense to use the initial balance for 24-hour scrips like bitcoin-USD pair or any other Forex pair. 

Insights and strength breakdown for the current day are displayed to the right and will get updated live. You can also hover the mouse over circles on top of market profile chart to see the insights (Those who are new to the plotly ecosystem, hover text is supported by plotly by default and Dash is the extension of Plotly).

Factors that are not contributing to the strength (where value = 0 in breakdown part) will get dynamically omitted to reduce the clutter and increase the readability. Also, when these factors are in play they will get automatically added in breakdown commentary (where breakdown value != 0).

## This is how the chart looks like in the new version
![image](https://user-images.githubusercontent.com/28746824/103477838-7d028280-4de8-11eb-9c5e-edbd436c3d92.png)


## General overview of the TPO chart produced by the algo (ignore the price as I copied this from readme section of the old versionThe)
![image](https://user-images.githubusercontent.com/28746824/103477858-a8856d00-4de8-11eb-9649-70d66a46b693.png)


## Bubbles at the top:
![How to read charts. Hover menu -2](https://user-images.githubusercontent.com/28746824/89723894-e341cf80-da19-11ea-84cd-a575f0a83bcc.png)

# Disclaimer about balanced target
if you read the insights and it's breakdown carefully there is a one line called a balanced target. So what is it and how important it is? well, I don't believe in using market profile as an indicator but thought to add this line as it indicates where the mean reversion target will be. This will be highly dangerous to use in scrips like Bitcoin or in general when the market is in strong trend. 

In the simplest form this is how it works. If there is an imbalance between POC -VAL and VAH-POC ( when price excessively moves to one direction) then it assumes that price will revert back and then it updates the balanced target. Don't take it too seriously, back then it used to work well in E-mini.

# Final thoughts

If you don't know anything about Market Profile (TM) then you might not understand the whole concept completely. If you're interested then do read the small handbook on Auction Market Theory(direct link below). Don't get confused by the name, AMT and Market Profile is the same thing. In short, the concept is based on how Auctions work. Of course the concept is not as deep as that 2020 Nobel Prize winner in Economics and quite frankly this one is quite usable. Usable in the sense most of the part is quantifiable. At the end of day it is a visual tool to assist the decision process and not an execution tool. In other words market profile is not an indicator or any kind of signal. 
# Link
CBOT Market Profile handbook (13 mb pdf)  https://t.co/L8DfNkLNi5?amp=1

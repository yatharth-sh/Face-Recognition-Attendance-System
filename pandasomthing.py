import csv
import pandas as pd
import os
import time
import datetime
 
df=pd.read_csv("data.csv")
def new__data():


    '''ID=1002
    Name="Ali"
    Attendance=True
    '''
    Date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")



    df=pd.read_csv("data.csv")
    new_row= {"ID":ID,"Name":Name,"TimeStamp":Date, "Attendance":Attendance}
    df= pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    #print(df) 
    return 

def show_peopledata():
    if not os.path.exists("data.csv"):
        print("File does not exists")
    elif df.empty:
        print("No data in the file yet")

    for i,row in df.iterrows():
        person=f"{row["ID"]},{row["Name"]},{row["Date"]},{row["Attendance"]}"
    
        print(person)
show_peopledata()


def face_recog():
    start=time.time()
    now=time.time()
    difference=now-start
    while True:
        user=input("Enter if want to start and q to quit: ")
        if user == '':
            new__data()
            start=time.time()
            now=time.time()
            difference=now-start
        elif user=='q' or difference>10:
            break


        
#print(df.head())
#print(df.columns)
#new__data()
#print(df) 
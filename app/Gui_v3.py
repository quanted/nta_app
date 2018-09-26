#!/d/home/apps/build64/bin/python

"""
 .. This GUI is meant to intialize the NTA Data Processing Packages
    Module Author: Hussein Al Ghoul hussein.al-ghoul@epa.gov
    Modified by: Jeffrey Minucci minucci.jeffrey@epa.gov on 9/11/18
"""
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import numpy
import os
import pandas as pd
import urllib.request, urllib.parse, urllib.error
#import glob
import functions_Universal_v3 as fn
from batch_search_v3 import BatchSearch
import Toxpi_v3 as tp
import CFMID_v3 as cfmid
#from multiprocessing import Process,Manager
#from Queue import Queue
from threading import Thread
import random
#from graphicHelp import help
import time

DIRECTORY = None
ERROR = 1
OUTPUT = ""


class Control(object):

    df=[0,0]
    dft = None
    Fname = ['','']
    Sname = ['','']
    Rname = ['','']
    name = ['','']
    Files = ['','']
    Tfile = ''
    entc2 = None
    ppm = None
    NegMGFfile = ''
    entm1 = None
    PosMGFfile = ''
    entm2 = None
    NegMGFtxtfile = ''
    entmgf1 = None
    PosMGFtxtfile = ''
    entmgf2 = None    
    
    df_cfmid = [None,None]
    def __int__(self, study = None, user = None):

        self.studyL = study
        self.userL = user
        self.search = None


    def study(self):
        st = ent0.get()
        self.studyL = st
    
    def user(self):
        us = ent1.get()
        self.userL = us


    def run(self):
    
        self.study()
        self.user()

    def load(self,con):

        self.studyL = con[0]
        self.userL = con[1]

        ent0.insert("end",con[0])
        ent1.insert("end",con[1])


    def check_file_exists(arg,f,index,csv=True):
        answer = ['','']
        file = ['','']
        file[index]=f
        FFPath = ['','']
        message = file[index] + " Exists. Would You Like to Replace It?"
        FFPath[index]=os.getcwd()+"/"+DIRECTORY+"/"+ file[index]
        timestr = time.strftime('%Y%m%d')
        if os.path.exists(FFPath[index]):
            if csv:
                file[index] = file[index].rsplit('.',1)[0] + "_" + timestr + ".csv"
            else:
                file[index] = file[index].rsplit('.',1)[0] + "_" + timestr + ".xlsx"                
        print(file[index])
        return file[index]


    def initialize(self):
        self.run()
        global DIRECTORY
        con = [self.studyL,self.userL]
        numpy.save(os.getcwd()+"/control_list.npy",con)
        if ERROR == 0:
            #frame2.grid(row=0,column=2,sticky="N"+"S"+"E"+"W")
            frame2.pack(side='left',expand=1,fill='both')
            btnrd.pack(side="left")
            ckbxrd2.pack(side='left')
            btndd.pack(side="left")
            ckbxrd3.pack(side='left')
            btnstat.pack(side="left")
            ckbxrd4.pack(side='left')
            btnct.pack(side="left")
            ckbxrd31.pack(side='left')
            btnbk1.pack(side="left")
            btnxt1.pack(side="left")
            root.update()
        if ERROR == 1:
            tkinter.messagebox.showinfo("ERROR","You Need to Select a Data File")
        DIRECTORY = con[0] + "_" + con[1]
        if not os.path.exists(DIRECTORY):
                os.makedirs(DIRECTORY)
        return DIRECTORY


    def openfile(self):
        global ERROR
        con = [self.studyL,self.userL]
        ent2.delete(0,tk.END)
        numpy.save(os.getcwd()+"/control_list.npy",con)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("tsv files","*.tsv"),("all files","*.*")))
        fname = ['','']
        for i in range(2):
            fname[i] = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname[0] +";"+fname[1]
        contra.Files = root.tk.splitlist(root.filename)            
        ent2.insert("end",fname_list)
        ERROR = 0
        return contra.Files
    #return root.filename


    def Tracers_openfile(self):
        global ERROR
        contra.entc2.delete(0,tk.END)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("tsv files","*.tsv"),("all files","*.*")))
        for i in range(1):
            fname = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname
        contra.tfile = root.tk.splitlist(root.filename)[0] 
        print(contra.tfile)           
        contra.entc2.insert("end",fname_list)
        ERROR = 0
        return contra.tfile
    


    def NegMGF_openfile(self):
        global ERROR
        contra.entm1.delete(0,tk.END)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("tsv files","*.tsv"),("all files","*.*")))
        for i in range(1):
            fname = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname
        contra.NegMGFfile = root.tk.splitlist(root.filename)[0] 
        print(contra.NegMGFfile)           
        contra.entm1.insert("end",fname_list)
        ERROR = 0
        return contra.NegMGFfile


    def PosMGF_openfile(self):
        global ERROR
        contra.entm2.delete(0,tk.END)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("tsv files","*.tsv"),("all files","*.*")))
        for i in range(1):
            fname = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname
        contra.PosMGFfile = root.tk.splitlist(root.filename)[0] 
        print(contra.PosMGFfile)           
        contra.entm2.insert("end",fname_list)
        ERROR = 0
        return contra.PosMGFfile



    def NegMGFtxt_openfile(self):
        global ERROR
        contra.entmgf1.delete(0,tk.END)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("mgf files","*.mgf"),("csv files","*.csv"),("all files","*.*")))
        for i in range(1):
            fname = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname
        contra.NegMGFtxtfile = root.tk.splitlist(root.filename)[0] 
        print(contra.NegMGFtxtfile)           
        contra.entmgf1.insert("end",fname_list)
        ERROR = 0
        return contra.NegMGFtxtfile


    def PosMGFtxt_openfile(self):
        global ERROR
        contra.entmgf2.delete(0,tk.END)
        root.filename = tkinter.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("mgf files","*.mgf"),("csv files","*.csv"),("all files","*.*")))
        for i in range(1):
            fname = root.tk.splitlist(root.filename)[i].rsplit('/',1)[-1]
        fname_list = fname
        contra.PosMGFtxtfile = root.tk.splitlist(root.filename)[0] 
        print(contra.PosMGFtxtfile)           
        contra.entmgf2.insert("end",fname_list)
        ERROR = 0
        return contra.PosMGFtxtfile
    
    

    def next1(self):
        frame3.pack(side='left',expand=1,fill='both')
        #frame3.grid(row=0,column=3,sticky="N"+"S"+"E"+"W")
        btncf.pack(side="left")
        ckbxrd32.pack(side='left')
        btncflags.pack(side="left")
        ckbxrd33.pack(side='left')
        btncd.pack(side="left")
        ckbxrd34.pack(side='left')
        btntp.pack(side="left")
        ckbxrd35.pack(side='left')
        btnbk2.pack(side="left")     
        btnxt2.pack(side="left")
        root.update()


    def next2(self):
        frame4.pack(side ='left',expand=1,fill='both')
        btnMGF.pack(side="left")
        ckbxrd40.pack(side="left")        
        btncfmidg.pack(side="left")
        ckbxrd41.pack(side="left")
        root.update()


    def back1(self):
        frame2.pack_forget()
        root.update()


    def back2(self):
        frame3.pack_forget()
        root.update()


    def Read_Data(self,arg,File,index):
        global OUTPUT
        print(checkCmd0.get())
        btnrd.configure(bg='light yellow')
        try:
            contra.df[index] = fn.read_data(File,index)
            if index == 1:
                OUTPUT = "Data was converted to Dataframe \n"
                btnrd.configure(bg='pale green')
            T.insert(tk.END, OUTPUT)
            return contra.df[index]
        except:
            print("This thread just failed")
            btnrd.configure(bg='IndianRed1')
            raise
            
            

    def Parse_Data(arg,index):
        global OUTPUT
        print(contra.df[index])
        contra.df[index] = fn.parse_headers(contra.df[index],index) # 3 corresponds to replicate number
        OUTPUT= "Data File was parsed! \n" 
        T.insert(tk.END, OUTPUT)
        return contra.df[index]      


    def Statistics(arg,File,index,mass_accuracy,rt_accuracy,ppm):
        global OUTPUT
        #mass_accuracy = 20
        #rt_accuracy = 0.05        
        btnstat.configure(bg='light goldenrod yellow')
        contra.Fname[index] = File.rsplit('/',1)[-1]
        contra.Sname[index] = contra.Fname[index].rsplit('.',1)[0] + "_Statistics.csv"
        try:           
            contra.df[index] = fn.statistics(contra.df[index],index)
            if varR1.get() == 1:
                OUTPUT ="Selected PPM \n"
                ppm=True
            else:
                OUTPUT ="Selected Da \n"
                ppm=False
            T.insert(tk.END, OUTPUT)
            contra.df[index] = fn.adduct_identifier(contra.df[index],index,float(mass_accuracy),float(rt_accuracy),ppm)
            if checkCmd2.get():
                OUTPUT = "Will Save the Statistics File \n"
                contra.name[index]=contra.check_file_exists(contra.Sname[index],index,csv=True)
                contra.df[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[index], index=False)
            else:
                OUTPUT = "Not Saving the Statistics File \n"
            T.insert(tk.END, OUTPUT)
            if index == 1:
                btnstat.configure(bg='pale green')
                OUTPUT = "Statistical Parsing is done! \n"
            return contra.df[index]      
        except:
            print("This thread just failed")
            btnstat.configure(bg='IndianRed1')
            raise

    def Check_Tracers(arg,File,index,mass_accuracy,rt_accuracy):

        #framec0.pack_forget()
        print(mass_accuracy)
        global OUTPUT
        btnct.configure(bg='light yellow')
        df_tracers = [None,None]
        contra.Fname[index] = File.rsplit('/',1)[-1]
        contra.Sname[index] = contra.Fname[index].rsplit('.',1)[0] + "_Tracers.csv"
        try:                        
            if varR1.get() == 1:
                OUTPUT ="Selected PPM \n"
                ppm=True
            else:
                OUTPUT ="Selected Da \n"
                ppm=False
            T.insert(tk.END, OUTPUT)
            df_tracers[index] = fn.check_feature_tracers(contra.df[index],str(contra.tfile),float(mass_accuracy),float(rt_accuracy),ppm)
            if checkCmd3.get():
                OUTPUT = "Will Save the Tracers File \n"
                contra.name[index]=contra.check_file_exists(contra.Sname[index],index,csv=True)
                df_tracers[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[index], index=False)
            else:
                OUTPUT = "Not Saving the Tracers File \n"
            T.insert(tk.END, OUTPUT)
            if index == 1:
                btnct.configure(bg='pale green')
                OUTPUT = "Tracers check is done! \n"
            T.insert(tk.END, OUTPUT)
            #return contra.df[index]      
        except:
            print("This thread just failed")
            btnct.configure(bg='IndianRed1')
            raise
            
            

    def Clean_Features(arg,File,index,controls):
        global OUPUT
        #controls = list()
        btncf.configure(bg='light yellow')
        contra.Fname[index] = File.rsplit('/',1)[-1]
        contra.Sname[index] = contra.Fname[index].rsplit('.',1)[0] + "_Clean.csv"
        if checkCmd0.get():
            ENTACT=True
            #controls = [3.0,2,1.5]
        else:
            ENTACT=False
            #controls = [3.0,2,0.5]
        print(controls)
        try:
            contra.df[index] = fn.clean_features(contra.df[index],index,ENTACT,controls)            
            if checkCmd4.get():
                OUTPUT = "Will Save the Cleaned File \n"
                contra.name[index]=contra.check_file_exists(contra.Sname[index],index,csv=True)
                contra.df[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[index], index=False)
            else:
                OUTPUT = "Not Saving the Cleaned File \n"
            T.insert(tk.END, OUTPUT)
            contra.df[index] = fn.Blank_Subtract(contra.df[index],index)
            contra.df[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.Fname[index].rsplit('.',1)[0]+"_reduced.csv", index=False)
            if index == 1:
                btncf.configure(bg='pale green')
                OUTPUT = "Features were cleaned! \n"
            T.insert(tk.END, OUTPUT)
            return contra.df[index] 
        except:
            print("This thread just failed")
            btncf.configure(bg='IndianRed1')
            raise

    def Create_Flags(arg,File,index):
        global OUPUT
        btncflags.configure(bg='light yellow')
        contra.Fname[index] = File.rsplit('/',1)[-1]
        contra.Sname[index] = contra.Fname[index].rsplit('.',1)[0] + "_Flags.csv"
        contra.Rname[index] = contra.Fname[index].rsplit('.',1)[0] + "_Reduced.csv"    

        try:
            contra.df[index] = fn.flags(contra.df[index])

            if checkCmd5.get():
                OUTPUT = "Will Save the Flags File \n"
                contra.name[index]=contra.check_file_exists(contra.Sname[index],index,csv=True)
                contra.df[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[index], index=False)
        #fn.reduce(contra.df[index],index).to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.Rname[index], index=False)
            else:
                OUTPUT = "Not Saving the Flags File \n"
            T.insert(tk.END, OUTPUT)
            if index == 1:
                btncflags.configure(bg='pale green')
                OUTPUT = "Flags were created! \n"
            T.insert(tk.END, OUTPUT)
            return contra.df[index] 
        except:
            print("This thread just failed")
            btncflags.configure(bg='IndianRed1')
            raise
        

    def Drop_Duplicates(arg,File,index):
        global OUPUT
        btndd.configure(bg='light yellow')
        contra.Fname[index] = File.rsplit('/',1)[-1]
        contra.Sname[index] = contra.Fname[index].rsplit('.',1)[0] + "_AfterDuplicates.csv"
        try:
            contra.df[index] = fn.duplicates(contra.df[index],index)
            if checkCmd1.get():
                OUTPUT = "Will Save the Duplicates File \n"
                contra.name[index]=contra.check_file_exists(contra.Sname[index],index,csv=True)
                contra.df[index].to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[index], index=False)
            else:
                OUTPUT = "Not Saving the Duplicates File \n"
            if index == 1:
                btndd.configure(bg='pale green')
                OUTPUT = "Duplicates were Removed! \n"
            T.insert(tk.END, OUTPUT)
            return contra.df[index]      
        except:
            print("This thread just failed")
            btndd.configure(bg='IndianRed1')
            raise
            
            

    def Combine_Modes(self):
        global OUPUT
        btncd.configure(bg='light yellow')
        #contra.Fname[0] = contra.Files[0].rsplit('/',1)[-1]
        contra.Sname[0] = "Data_Both_Modes_Combined.csv"
        try:
            contra.dft = fn.combine(contra.df[0],contra.df[1])
            if checkCmd6.get():
                OUTPUT = "Will Save the Combined File \n"
                contra.name[0]=contra.check_file_exists(contra.Sname[0],0,csv=True)
                contra.dft.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[0], index=False)
            else:
                OUTPUT = "Not Saving the Combined File \n"
            btncd.configure(bg='pale green')
            directory = os.getcwd()+"/"+DIRECTORY
            fn.MPP_Ready(contra.dft,directory,contra.Fname[0].rsplit('.',1)[0])
            T.insert(tk.END, OUTPUT)
            OUTPUT = "DModes were Combined! \n"
            T.insert(tk.END, OUTPUT)
        except:
            print("This thread just failed")
            btncd.configure(bg='IndianRed1')
            raise

    def Tox_Pi(self):
        global OUPUT
        btntp.configure(bg='light yellow')
        contra.DB_Controls()
        btndb.wait_variable(vardb1)
        top4.withdraw()
                
        try:
            if varRdb1.get() == 1:
                OUTPUT ="Searching by Mass \n"
                by_mass = True
                by_formula = False
            else:
                OUTPUT ="Searching by Formula \n"
                by_mass = False
                by_formula = True            
            T.insert(tk.END, OUTPUT)                
            if varRdb2.get() == 1:
                OUTPUT ="Saving Top Dashboard Hit \n"
                tophit=True
            else:
                OUTPUT ="Selected All Dashboard Hit \n"
                tophit=False
            T.insert(tk.END, OUTPUT)  
            if by_formula:  
                contra.Batch_Search(by_mass=False)
            if by_mass:
                contra.Batch_Search(by_mass=True, ppm=wdb.get())
            directory = os.getcwd()+"/"+DIRECTORY
            #contra.Fname[0] = contra.Files[0].rsplit('/',1)[-1]
            #contra.Sname[0] = "Data_Both_Modes_toxpi.xlsx"
            contra.Sname[0] = "Data_Both_Modes_toxpi.csv"
            
            print("Finished the Selenium part")
            time.sleep(5)
            #dashboard_file[0] = [filename for filename in os.listdir(os.getcwd()+"/"+DIRECTORY) if filename.startswith('ChemistryDashboard-AdvancedSearch')]
            dashboard_file = contra.Download_Finished()
            print("This is the dashboard_file: " + dashboard_file)
            contra.dft = tp.process_toxpi(contra.dft,directory, dashboard_file,tophit,by_mass)
            '''the following line is commented out until the toxpi calculation methond are reviewed'''
            #contra.dft = tp.calculate_toxpi(contra.dft,directory) 
            if checkCmd7.get():
                OUTPUT = "Will Save the ToxPi File \n"
                contra.name[0]=contra.check_file_exists(contra.Sname[0],0,csv=False)
                contra.dft.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[0], index=False)
                #contra.dft.to_excel(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[0],na_rep='',engine='xlsxwriter')
                #contra.dft.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[0], index=False)
            else:
                OUTPUT = "Not Saving the toxpi File \n"
            btntp.configure(bg='pale green')
            T.insert(tk.END, OUTPUT)
            OUTPUT = "Toxpi Data was Created! \n"
            T.insert(tk.END, OUTPUT)
            #contra.dft.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+"toxpi_merged_file.csv",index=False)
            return contra.dft
        except:
            print("This thread just failed")
            btntp.configure(bg='IndianRed1')
            raise

    def CFMID_Query(arg,mode,mgf_file,index):
        btncfmidg.configure(bg='light yellow')        
        cfmid_ppm = float(entcd0.get())
        cfmid_ppm_sl = float(entcd1.get())
        try:
            if varRcd3.get():
                filtering = True
            else:
                filtering = False
            if mode == 'positive':
                POSMODE = True
            if mode == 'negative':
                POSMODE = False            
            dfpc = contra.dft.reset_index()
            contra.df_cfmid[index] = cfmid.compare_mgf_df(mgf_file,dfpc,cfmid_ppm,cfmid_ppm_sl,POSMODE,filtering)
            print(contra.df_cfmid[index])
        except:
            print("This thread just failed")
            btncfmidg.configure(bg='IndianRed1')
            raise

    def CFMID_Combine(self):
        df_cfmid_neg_AE =  contra.df_cfmid[0][0]
        df_cfmid_neg_S =  contra.df_cfmid[0][1]
        df_cfmid_pos_AE =  contra.df_cfmid[1][0]
        df_cfmid_pos_S =  contra.df_cfmid[1][1]
        df_cfmid_AE = pd.concat([df_cfmid_neg_AE,df_cfmid_pos_AE])
        df_cfmid_S = pd.concat([df_cfmid_neg_S,df_cfmid_pos_S])
        df_cfmid_AE_merged = cfmid.merge_pcdl(contra.dft.reset_index(),df_cfmid_AE)
        df_cfmid_S_merged = cfmid.merge_pcdl(contra.dft.reset_index(),df_cfmid_S)        
        try:
            if checkCmd8.get():
                OUTPUT = "Will Save the CFMID Files \n"
                df_cfmid_AE_merged.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+'Data_MultiScores_Both_Modes.csv',index=False)
                df_cfmid_S_merged.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+'Data_OneScores_Both_Modes.csv',index=False)
                
                #df_cfmid_AE_merged.to_excel(os.getcwd()+"/"+DIRECTORY+"/"+'Data_MultiScores_Both_Modes.xlsx',na_rep='',engine='xlsxwriter')
                #df_cfmid_S_merged.to_excel(os.getcwd()+"/"+DIRECTORY+"/"+'Data_OneScore_Both_Modes.xlsx',na_rep='',engine='xlsxwriter')                
                #contra.dft.to_csv(os.getcwd()+"/"+DIRECTORY+"/"+contra.name[0], index=False)
            else:
                OUTPUT = "Not Saving the toxpi File \n"  
            T.insert(tk.END, OUTPUT)

        except:
            print("This thread just failed")
            btncfmidg.configure(bg='IndianRed1')
            raise            

    def MGF_Converter(arg,mgf_tfile,index):
        btnMGF.configure(bg='light yellow')        
        try:
            print(mgf_tfile.split('.',1)[0]+'.csv')
            if os.path.isfile(mgf_tfile.split('.',1)[0]+'.csv'):
                OUTPUT = "This file already exists, not recreating it."
                T.insert(tk.END, OUTPUT)    
            else:
                cfmid.parseMGF(mgf_tfile)
        except:
            print("This thread just failed")
            btnMGF.configure(bg='IndianRed1')
            raise           


          
    def plot_window(self):
        plotw = tk.Toplevel()
        plotw.title("ToxPi")
        btnplot=tk.Button(root,text="Plot" , height=2 , width = 15,command = lambda: contra.plot())



    def Batch_Search(self,by_mass=True,ppm=10):
        dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DIRECTORY)
        compounds = list()
        mono_masses = list()
        if by_mass:
            mono_masses = fn.masses(contra.dft)
            mono_masses_str = [str(i) for i in mono_masses]
            self.search = BatchSearch()
            self.search.batch_search(masses=mono_masses_str,formulas=None,directory=dir,by_formula=False,ppm=ppm)
        else:
            compounds = fn.formulas(contra.dft)
            self.search = BatchSearch()
            self.search.batch_search(masses=None, formulas=compounds, directory=dir)

    #bs.batch_search(compounds,directory)

    def Download_Finished(self):
        directory = os.getcwd()+"/"+DIRECTORY
        finished = False
        file_list = []
        for i in range(100):
            for filename in os.listdir(directory):
                if filename.startswith('ChemistryDashboard-Batch-Search'):
                    #print("in loop filename: "+filename)
                    if filename not in file_list and not filename.endswith("part"):
                        file_list.append(filename)
                        finished = True
            if finished and i>10: #if there are multiple, wait 10 secs to see if a new one is downloaded
                break
            time.sleep(1)
        if not finished:
            raise Exception("Download from the CompTox Chemistry Dashboard failed!")
        if len(file_list) > 1:
            print("Multiple downloads found: "+ str(file_list))
            print("Using the last one.")
        file = file_list[len(file_list)-1]
        print("This is what was downloaded: " + file)
        self.search.close_driver()
        return file
    

# Multiprocessing Functions

    def RD_MP(self):
        threads = []    
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Read_Data,args=(self, f,index))
            threads.append(thread)
        #thread.start()
        for thread in threads:
            thread.start()



    def DD_MP(self):
        threads = []
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Drop_Duplicates,args=(f,index))
            threads.append(thread)
        for thread in threads:
            thread.start()


    def S_MP(self):
        adduct=True
        threads = []
        contra.Accuracies(adduct)
        btnac.wait_variable(var)
        top2.withdraw()
        if entc0.get():
            mass_accuracy = entc0.get()
            rt_accuracy = entc1.get()
        else:
            mass_accuracy = 20
            rt_accuracy = 0.5
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Statistics,args=(f,index,mass_accuracy,rt_accuracy,contra.ppm))
            threads.append(thread)
        for thread in threads:
            thread.start()

            


    def CF_MP(self):
        threads = []
        controls = list()
        rep = fn.Replicates_Number(contra.df[0],0)
        print(rep)
        contra.Controls_Win(rep)
        contra.Controls()
        btncut.wait_variable(var1)
        top3.withdraw()
        controls.append(float(entct0.get()))
        controls.append(w.get())
        controls.append(float(entct1.get()))
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Clean_Features,args=(f,index,controls))
            threads.append(thread)
        for thread in threads:
            thread.start()


    def F_MP(self):
        threads = []
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Create_Flags,args=(f,index))
            threads.append(thread)
        for thread in threads:
            thread.start()


    def CT_MP(self):
        threads = []
        adduct=False
        contra.Accuracies(adduct)
        btnac.wait_variable(var)
        top2.withdraw()
        if entc0.get() != None :
            mass_accuracy = entc0.get()
            rt_accuracy = entc1.get()
        else:
            mass_accuracy = 20
            rt_accuracy = 0.5
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Check_Tracers,args=(f,index,mass_accuracy,rt_accuracy))
            threads.append(thread)
        for thread in threads:
            thread.start()


    def BS_MP(self):
        threads = []
        for index, f in enumerate(contra.Files):
            thread = Thread(target=contra.Blank_Subtract,args=(f,index))
            threads.append(thread)
        for thread in threads:
            thread.start()


    def MGF_MP(self):
        threads = []
        contra.MGF_Controls()
        btnmgf.wait_variable(varmgf)
        top6.withdraw()
        btnMGF.configure(bg='light yellow')
        files = list()
        files.append(contra.NegMGFtxtfile)
        files.append(contra.PosMGFtxtfile)
        for index, f in enumerate(files):
            thread = Thread(target=contra.MGF_Converter,args=(f,index))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        #btnMGF.configure(bg='pale green')
        OUTPUT = "Parsed MGF Files."
        T.insert(tk.END, OUTPUT)  
        btnMGF.configure(bg='pale green')
        ''' make the mgf_converter function'''


    def CFMID_MP(self):
        threads = []
        contra.CFMID_Controls()
        btncfmid.wait_variable(varcd)
        top5.withdraw()
        files = list()
        files.append(contra.NegMGFfile)
        files.append(contra.PosMGFfile)
        modes = ['negative','positive']
        for i in range(len(files)):
            thread = Thread(target=contra.CFMID_Query,args=(modes[i],files[i],i))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        contra.CFMID_Combine()
        btncfmidg.configure(bg='pale green')
        OUTPUT = "CFMID Data was Created! \n"
        T.insert(tk.END, OUTPUT)
        print("this is what cfmid would do") 



    def Accuracies(self,adduct):
        if adduct:
            w = 320
            h = 170
            dx = 400
            dy = 400
            top2.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
            top2.deiconify()            
            framec0.config(text='Adduct Controls',foreground='black')
            btnac.pack(padx=5,pady=10,expand=1,side='top')
        else:
            w = 360
            h = 240
            dx = 400
            dy = 400
            top2.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
            top2.deiconify()
            framec0.config(text='Tracers Controls',foreground='black')

            lc2=tk.Label(framec02, text="Tracers File",foreground='black')
            lc2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
            contra.entc2=tk.Entry(framec02, width=22, font=15)
            contra.entc2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
            btncfile=tk.Button(framec02, text="Open",  height=1 , width = 5,command = lambda: contra.Tracers_openfile())
            btncfile.pack(padx=5,pady=15,expand=0,side="left",fill='both')  
            btnac.pack(padx=5,pady=10,expand=1,side='top')
        framec0.pack(side='bottom',padx=10,expand=1,fill='both')

    def Controls(self):
        h = top3.winfo_width()
        w = top3.winfo_height()
        dx = 300
        dy = 400
        top3.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
        top3.deiconify()
        
        framec1.pack(side='bottom',padx=10,pady=10,expand=1,fill='both')   
        
    def Controls_Win(arg,replicate_number):
        #controls for cutoffs to be entered
        w.config(to=replicate_number)
 


    def DB_Controls(self):
        h = top4.winfo_width()
        w = top4.winfo_height()
        dx = 300
        dy = 400
        top4.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
        top4.deiconify()
        
        framed1.pack(side='bottom',padx=10,pady=10,expand=1,fill='both') 


    def CFMID_Controls(self):
        entcd0.insert(0,wdb.get())
        #varRcd1.set(1)
        h = top5.winfo_width()
        w = top5.winfo_height()
        dx = 350
        dy = 450
        top5.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
        top5.deiconify()
        
        framecd0.pack(side='bottom',padx=10,pady=10,expand=1,fill='both') 


    def MGF_Controls(self):
        h = top6.winfo_width()
        w = top6.winfo_height()
        dx = 400
        dy = 220
        top6.geometry("%dx%d+%d+%d" % (w,h,x+dx,y+dy))
        top6.deiconify()
        
        framemgf0.pack(side='bottom',padx=10,pady=10,expand=1,fill='both') 

    def ask_quit(self):
        message = ["Later."]
        if tkinter.messagebox.askokcancel("Quit", random.choice(message)):
            top1.destroy()
            rootS.destroy()
            root.destroy()
            hidden0.destroy()
            hidden1.destroy()
            
lst = ['STUDY','USER','DATA']



# Main GUI window
contra = Control()
root = tk.Tk()
top1 = tk.Toplevel()
top1.wm_title("NTA ANALYZER")
top2 = tk.Toplevel()
top2.wm_title("ACCURACY CONTROLS")
top2.withdraw()
top3 = tk.Toplevel()
top3.wm_title("CUTTOFF CONTROLS")
top3.wm_geometry("220x300")
top3.withdraw()
top4 = tk.Toplevel()
top4.wm_title("DASHBOARD SEARCH CONTROLS")
top4.wm_geometry("230x340")
top4.withdraw()
top5 = tk.Toplevel()
top5.wm_title("CFMID MATCHING CONTROLS")
top5.wm_geometry("350x430")
top5.withdraw()
top6 = tk.Toplevel()
top6.wm_title("MGF CONVERTING CONTROLS")
top6.wm_geometry("220x440")
top6.withdraw()

root.withdraw()
x = root.winfo_x()
y = root.winfo_y()

top1.group(root)
frame=tk.LabelFrame(top1, text = 'Details' ,height = 600, width = 600)
frame2=tk.LabelFrame(top1, text = 'Step One' ,height = 600, width = 600)
frame3=tk.LabelFrame(top1, text = 'Step Two' ,height = 600, width = 600)
frame4=tk.LabelFrame(top1, text = 'Step Three' ,height = 600, width = 600)



frame.pack(side='left',padx=10,expand=1,fill='both')


frame00=tk.Frame(frame)
frame00.pack(expand=1,fill='both')
l0=tk.Label(frame00, text=lst[0])
l0.pack(padx=5,pady=15,expand=1,side="left",fill='both')
ent0=tk.Entry(frame00, width=30, font=55)
ent0.pack(padx=5,pady=15,expand=1,side="left",fill='both')


frame01=tk.Frame(frame)
frame01.pack(expand=1,fill='both')
l1=tk.Label(frame01, text=lst[1])
l1.pack(padx=10,pady=15,expand=1,side="left",fill='both')
ent1=tk.Entry(frame01, width=30, font=55)
ent1.pack(padx=5,pady=15,expand=1,side="left",fill='both')


frame02=tk.Frame(frame)
frame02.pack(expand=1,fill='both')
l2=tk.Label(frame02, text=lst[2])
l2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
ent2=tk.Entry(frame02, width=22, font=15)
ent2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
btnfile=tk.Button(frame02, text="Open",  height=1 , width = 5,command = lambda: contra.openfile())
btnfile.pack(padx=5,pady=15,expand=0,side="left",fill='both')



frame1=tk.Frame(frame)
frame1.pack(expand=1,side="top")

btnSt=tk.Button(frame1, text="Initialize" , command= lambda: contra.initialize())
btnSt.pack(expand=1,side='top')





####first frame where binning is done#####
checkCmd0 = tk.IntVar()
checkCmd1 = tk.IntVar()
checkCmd2 = tk.IntVar()
checkCmd3 = tk.IntVar()

frame21=tk.Frame(frame2)
frame21.pack(padx=80,expand=1,fill='both')
btnrd=tk.Button(frame21, text="Read Data" ,  height=2 , width = 15,command = lambda: contra.RD_MP())
ckbxrd2=tk.Checkbutton(frame21,variable=checkCmd0 ,text="ENTACT")

frame22=tk.Frame(frame2)
frame22.pack(padx=80,expand=1,fill='both')
btndd=tk.Button(frame22,text="Drop Duplicates" , height=2 , width = 15,command = lambda: contra.DD_MP())
ckbxrd3=tk.Checkbutton(frame22,variable=checkCmd1 ,text="Save File")

frame23=tk.Frame(frame2)
frame23.pack(padx=80,expand=1,fill='both')
btnstat=tk.Button(frame23, text="Statistics" ,  height=2 , width = 15,command= lambda: contra.S_MP())
ckbxrd4=tk.Checkbutton(frame23,variable=checkCmd2 ,text="Save File")


frame24=tk.Frame(frame2)
frame24.pack(padx=80,expand=1,fill='both')
btnct=tk.Button(frame24, text="Tracers Check" ,  height=2 , width = 15,command= lambda: contra.CT_MP())
ckbxrd31=tk.Checkbutton(frame24,variable=checkCmd3 ,text="Save File")



btnxt1=tk.Button(frame2, text="Next",  height=2 , width = 15,command = lambda: contra.next1())
btnbk1=tk.Button(frame2, text="Back", height=2 , width = 15 ,command = lambda: contra.back1())



####second frame where Decay Amps are calculated####

checkCmd4 = tk.IntVar()
checkCmd5 = tk.IntVar()
checkCmd6 = tk.IntVar()
checkCmd7 = tk.IntVar()

frame31=tk.Frame(frame3)
frame31.pack(padx=80,expand=1,fill='both')
btncf=tk.Button(frame31,text="Clean Features" , height=2 , width = 15,command = lambda: contra.CF_MP())
ckbxrd32=tk.Checkbutton(frame31,variable=checkCmd4 ,text="Save File")

frame32=tk.Frame(frame3)
frame32.pack(padx=80,expand=1,fill='both')
btncflags=tk.Button(frame32, text="Create Flags", height=2, width =15, command = lambda: contra.F_MP())
ckbxrd33=tk.Checkbutton(frame32, variable=checkCmd5 ,text="Save File")

frame33=tk.Frame(frame3)
frame33.pack(padx=80,expand=1,fill='both')
btncd=tk.Button(frame33, text="Combine Modes" ,  height=2 , width = 15,command= lambda: contra.Combine_Modes())
ckbxrd34=tk.Checkbutton(frame33,variable=checkCmd6 ,text="Save File")

frame34=tk.Frame(frame3)
frame34.pack(padx=80,expand=1,fill='both')
btntp=tk.Button(frame34, text="Dashboard Search" ,  height=2 , width = 15,command= lambda: contra.Tox_Pi())
ckbxrd35=tk.Checkbutton(frame34,variable=checkCmd7 ,text="Save File")


btnxt2=tk.Button(frame3, text="Next",  height=2 , width = 15,command = lambda: contra.next2())
btnbk2=tk.Button(frame3, text="Back", height=2 , width = 15 ,command = lambda: contra.back2())


### last window Where CFMID is done

checkCmd8 = tk.IntVar()
checkCmd9 = tk.IntVar()

frame41=tk.Frame(frame4)
frame41.pack(padx=80,expand=1,fill='both')
btnMGF=tk.Button(frame41,text='Convert MGF' , height=2 , width = 15,command = lambda: contra.MGF_MP())
ckbxrd40=tk.Checkbutton(frame41,variable=checkCmd8 ,text="Save File")

frame42=tk.Frame(frame4)
frame42.pack(padx=80,expand=1,fill='both')
btncfmidg=tk.Button(frame42,text="CFMID" , height=2 , width = 15,command = lambda: contra.CFMID_MP())
ckbxrd41=tk.Checkbutton(frame42,variable=checkCmd9 ,text="Save File")




#controls for Accuracies to be entered
framec0=tk.LabelFrame(top2 ,height = 300, width = 400)
frameTracers0 = tk.Frame(framec0)
frameTracers0.pack(expand=1,fill='both')
lc0=tk.Label(frameTracers0, text='Mass Accuracy',foreground='black')
lc0.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varR1 = tk.IntVar()
#varR2 = tk.IntVar()
R1 = tk.Radiobutton(frameTracers0, text="ppm",foreground='black', variable=varR1, value=1)
R1.pack(padx=2,pady=15,expand=1,side="left",fill='both')
R2 = tk.Radiobutton(frameTracers0, text="Da",foreground='black', variable=varR1, value=0)
R2.pack(padx=2,pady=15,expand=1,side="left",fill='both')
entc0=tk.Entry(frameTracers0, width=5, font=55)
entc0.pack(padx=2,pady=10,expand=0,side="left",fill='both')


frameTracers1 = tk.Frame(framec0)
frameTracers1.pack(expand=1,fill='both')
lc1=tk.Label(frameTracers1, text='RT Accuracy (min)',foreground='black')
lc1.pack(padx=5,pady=15,expand=1,side="left",fill='both')
entc1=tk.Entry(frameTracers1, width=5, font=55)
entc1.pack(padx=2,pady=10,expand=0,side="left",fill='both')
framec02=tk.Frame(framec0)
framec02.pack(expand=1,fill='both')
var = tk.IntVar()
btnac=tk.Button(framec0, text="Use" ,height=2, width =5, command= lambda: var.set(1))

#btnac=tk.Button(framec0, text="Use" , command= lambda: framec0.pack_forget())

''' This block is for the feature cleaning inputs window '''

framec1=tk.LabelFrame(top3, text = 'Cutoffs',foreground='black' ,height = 600, width = 600)
frameCut1 = tk.Frame(framec1)
frameCut1.pack(expand=1,fill='both')
lct0=tk.Label(frameCut1, text='Sample To Blank Ratio',foreground='black')
lct0.pack(padx=5,pady=5,expand=1,side="left",fill='both')
entct0=tk.Entry(frameCut1, width=5, font=55)
entct0.pack(padx=20,pady=15,expand=0,side="left",fill='both')

frameCut2 = tk.Frame(framec1)
frameCut2.pack(expand=1,fill='both')
lct1=tk.Label(frameCut2, text='Minimum Replicate Hits',foreground='black')
lct1.pack(padx=5,pady=15,expand=1,side="left",fill='both')
w = tk.Scale(frameCut2, from_=1,resolution=1, orient='horizontal')
w.pack(padx=5,pady=5,expand=1,side="left",fill='both')


frameCut3 = tk.Frame(framec1)
frameCut3.pack(expand=1,fill='both')
lct2=tk.Label(frameCut3, text='Maximum Replicate CV',foreground='black')
lct2.pack(padx=5,pady=5,expand=1,side="left",fill='both')
entct1=tk.Entry(frameCut3, width=5, font=55)
entct1.pack(padx=20,pady=15,expand=0,side="left",fill='both')

var1 = tk.IntVar()
btncut=tk.Button(framec1, text="Use" , command= lambda: var1.set(1))
#btnac=tk.Button(framec0, text="Use" , command= lambda: framec0.pack_forget())
btncut.pack(expand=1,side='top')



''' This block is for the dashboard search inputs window '''

framed1=tk.LabelFrame(top4, text = 'Controls',foreground="black" ,height = 600, width = 600)
framedb1 = tk.Frame(framed1)
framedb1.pack(expand=1,fill='both')
ldt0=tk.Label(framedb1, text='Parent Ion Mass Accuracy PPM',foreground='black')
ldt0.pack(padx=5,pady=5,expand=1,side="left",fill='both')
wdb = tk.Scale(framedb1, from_=1,to=10,resolution=1, orient='horizontal')
wdb.pack(padx=5,pady=5,expand=1,side="left",fill='both')


framedb2 = tk.Frame(framed1)
framedb2.pack(expand=1,fill='both')
ldt1=tk.Label(framedb2, text='Search Dashboard by:',foreground='black')
ldt1.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varRdb1 = tk.IntVar()
#varR2 = tk.IntVar()
Rdb11 = tk.Radiobutton(framedb2, text="Mass",foreground='black', variable=varRdb1, value=1)
Rdb11.pack(padx=2,pady=15,expand=1,side="left",fill='both')
Rdb12 = tk.Radiobutton(framedb2, text="Formula",foreground='black', variable=varRdb1, value=0)
varRdb1.set(1)

Rdb12.pack(padx=2,pady=15,expand=1,side="left",fill='both')


framedb3 = tk.Frame(framed1)
framedb3.pack(expand=1,fill='both')
ldt2=tk.Label(framedb3, text='Save Top Dashboard Result Only?',foreground='black')
ldt2.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varRdb2 = tk.IntVar()
#varR2 = tk.IntVar()
Rdb21 = tk.Radiobutton(framedb3, text="Yes",foreground='black', variable=varRdb2, value=1)
Rdb21.pack(padx=2,pady=15,expand=1,side="left",fill='both')
Rdb22 = tk.Radiobutton(framedb3, text="No",foreground='black', variable=varRdb2, value=0)
Rdb22.pack(padx=2,pady=15,expand=1,side="left",fill='both')



vardb1 = tk.IntVar()
btndb=tk.Button(framed1, text="Use" , command= lambda: vardb1.set(1))
#btnac=tk.Button(framec0, text="Use" , command= lambda: framec0.pack_forget())
btndb.pack(expand=1,side='top')



# CFMID windows controls
framecd0=tk.LabelFrame(top5 ,height = 300, width = 400)
framecfmid0 = tk.Frame(framecd0)
framecfmid0.pack(expand=1,fill='both')
lcd0=tk.Label(framecfmid0, text='Parent Mass Accuracy')
lcd0.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varRcd1 = tk.IntVar()
#varR2 = tk.IntVar()
Rcd11 = tk.Radiobutton(framecfmid0, text="ppm",foreground='black', variable=varRcd1, value=1)
Rcd11.pack(padx=2,pady=15,expand=1,side="left",fill='both')
Rcd12 = tk.Radiobutton(framecfmid0, text="Da",foreground='black', variable=varRcd1, value=0)
Rcd12.pack(padx=2,pady=15,expand=1,side="left",fill='both')
entcd0=tk.Entry(framecfmid0, width=5, font=55)
entcd0.pack(padx=2,pady=10,expand=0,side="left",fill='both')

framecfmid1 = tk.Frame(framecd0)
framecfmid1.pack(expand=1,fill='both')
lcd1=tk.Label(framecfmid1, text='Fragment Mass Accuracy')
lcd1.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varRcd2 = tk.IntVar()
Rcd21 = tk.Radiobutton(framecfmid1, text="ppm",foreground='black', variable=varRcd2, value=1)
Rcd21.pack(padx=2,pady=15,expand=1,side="left",fill='both')
Rcd22 = tk.Radiobutton(framecfmid1, text="Da",foreground='black', variable=varRcd2, value=0)
Rcd22.pack(padx=2,pady=15,expand=1,side="left",fill='both')
entcd1=tk.Entry(framecfmid1, width=5, font=55)
entcd1.pack(padx=2,pady=10,expand=0,side="left",fill='both')

framecfmid3 = tk.Frame(framecd0)
framecfmid3.pack(expand=1,fill='both')
lcdt1=tk.Label(framecfmid3, text='Use CFMID Filtering?',foreground='black')
lcdt1.pack(padx=5,pady=10,expand=1,side="left",fill='both')
varRcd3 = tk.IntVar()
#varR2 = tk.IntVar()
Rcd31 = tk.Radiobutton(framecfmid3, text="Yes",foreground='black', variable=varRcd3, value=1)
Rcd31.pack(padx=2,pady=15,expand=1,side="left",fill='both')
Rcd32 = tk.Radiobutton(framecfmid3, text="No",foreground='black', variable=varRcd3, value=0)
Rcd32.pack(padx=2,pady=15,expand=1,side="left",fill='both')

framecfmid4 = tk.Frame(framecd0)
framecfmid4.pack(expand=1,fill='both')
lcd3=tk.Label(framecfmid4, text="Negative Mode MGF File",foreground='black')
lcd3.pack(padx=5,pady=15,expand=1,side="left",fill='both')
contra.entm1=tk.Entry(framecfmid4, width=22, font=15)
contra.entm1.pack(padx=5,pady=15,expand=1,side="left",fill='both')
btnm1file=tk.Button(framecfmid4, text="Open",  height=1 , width = 5,command = lambda: contra.NegMGF_openfile())
btnm1file.pack(padx=5,pady=15,expand=0,side="left",fill='both') 
'''varm1 = tk.IntVar()
btnm1=tk.Button(framecfmid4, text="Use" ,height=2, width =5, command= lambda: varm1.set(1)) 
btnm1.pack(padx=5,pady=10,expand=1,side='top')
'''

framecfmid5 = tk.Frame(framecd0)
framecfmid5.pack(expand=1,fill='both')
lcd5=tk.Label(framecfmid5, text="Positive Mode MGF File",foreground='black')
lcd5.pack(padx=5,pady=15,expand=1,side="left",fill='both')
contra.entm2=tk.Entry(framecfmid5, width=22, font=15)
contra.entm2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
btnm2file=tk.Button(framecfmid5, text="Open",  height=1 , width = 5,command = lambda: contra.PosMGF_openfile())
btnm2file.pack(padx=5,pady=15,expand=0,side="left",fill='both')
'''varm2 = tk.IntVar()
btnm2=tk.Button(framecfmid5, text="Use" ,height=2, width =5, command= lambda: varm2.set(1))   
btnm2.pack(padx=5,pady=10,expand=1,side='top')
'''

framecfmid6=tk.Frame(framecd0)
framecfmid6.pack(expand=1,fill='both')
varcd = tk.IntVar()
btncfmid=tk.Button(framecfmid6, text="Use" ,height=2, width =5, command= lambda: varcd.set(1))
btncfmid.pack(expand=1,side='top')



# MGF converter window
framemgf0=tk.LabelFrame(top6 ,height = 300, width = 400)

framemgf1 = tk.Frame(framemgf0)
framemgf1.pack(expand=1,fill='both')
lmg3=tk.Label(framemgf1, text="Negative Mode MGF File",foreground='black')
lmg3.pack(padx=5,pady=15,expand=1,side="left",fill='both')
contra.entmgf1=tk.Entry(framemgf1, width=22, font=15)
contra.entmgf1.pack(padx=5,pady=15,expand=1,side="left",fill='both')
btnmgf1file=tk.Button(framemgf1, text="Open",  height=1 , width = 5,command = lambda: contra.NegMGFtxt_openfile())
btnmgf1file.pack(padx=5,pady=15,expand=0,side="left",fill='both') 



framemgf2 = tk.Frame(framemgf0)
framemgf2.pack(expand=1,fill='both')
lcd5=tk.Label(framemgf2, text="Positive Mode MGF File",foreground='black')
lcd5.pack(padx=5,pady=15,expand=1,side="left",fill='both')
contra.entmgf2=tk.Entry(framemgf2, width=22, font=15)
contra.entmgf2.pack(padx=5,pady=15,expand=1,side="left",fill='both')
btnm2file=tk.Button(framemgf2, text="Open",  height=1 , width = 5,command = lambda: contra.PosMGFtxt_openfile())
btnm2file.pack(padx=5,pady=15,expand=0,side="left",fill='both')


framemgf3 = tk.Frame(framemgf0)
framemgf3.pack(expand=1,fill='both')
varmgf = tk.IntVar()
btnmgf=tk.Button(framemgf3, text="Use" ,height=2, width =5, command= lambda: varmgf.set(1))
btnmgf.pack(expand=1,side='top')




# Output Message log Window
rootS = tk.Toplevel()
rootS.title("MESSAGE LOG")
S = tk.Scrollbar(rootS)
T = tk.Text(rootS , height=10, width=50)
S.pack(side='right', fill='both')
T.pack(side='left', fill='both')
S.config(command = T.yview)
T.config(yscrollcommand=S.set)
'''wm=300
hm=300
ws = rootS.winfo_screenwidth()
hs = rootS.winfo_screenheight()
xm = 0.45*ws
ym = 0.5*hs'''
#rootS.geometry('%dx%d+%d+%d' % (wm,hm,xm,ym))

hidden0 = tk.Toplevel()
hidden0.withdraw()

hidden1 = tk.Toplevel()
hidden1.withdraw()


if os.path.isfile(os.getcwd()+"/control_list.npy"):
    con = numpy.load(os.getcwd()+"/control_list.npy")
    contra.load(con)

#   print "Hi there"
top1.protocol("WM_DELETE_WINDOW", contra.ask_quit)
root.mainloop()
#rootS.mainloop()


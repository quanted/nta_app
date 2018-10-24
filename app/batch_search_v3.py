from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.remote.remote_connection import LOGGER

import logging
import os
import time
import urllib.request, urllib.parse, urllib.error

driver = None

class BatchSearch:

    def __init__(self,bit_64 = True):
        self.bit_64 = bit_64
        self.driver = None

    def batch_search(self,masses=None,formulas=None,directory='',by_formula=True,ppm=10):
        # chrome_profile = webdriver.ChromeOptions()
        # profile = {"download.default_directory": directory,
        #        "download.prompt_for_download": False,
        #        "download.directory_upgrade": True,
        #        "safebrowsing.enabled": True}

        #chrome_profile.add_experimental_option("prefs", profile)
        LOGGER.setLevel(logging.WARNING)
        options = webdriver.firefox.options.Options()
        options.set_headless(headless=True)  # change this to false to see the browser in action (slower)
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference("browser.download.dir",directory)
        firefox_profile.set_preference("browser.download.folderList",2)
        firefox_profile.set_preference("browser.download.manager.showWhenStarting",False)
        firefox_profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/tab-separated-values")
    
        external_url = "https://comptox.epa.gov/dashboard/dsstoxdb/batch_search"
        internal_url = "http://comptox.zn.epa.gov/dashboard/dsstoxdb/batch_search"
        #Helpful command line switches
        # http://peter.sh/experiments/chromium-command-line-switches/
    
        #chrome_profile.add_argument("--disable-extensions")
        #self.driver = webdriver.Chrome(executable_path=os.getcwd()+"\chromedriver.exe",
        #chrome_options=chrome_profile)
        gecko_dir = os.path.dirname(os.path.abspath(__file__))
        gecko_path = os.path.join(gecko_dir,"geckodriver.elf")
        # if self.bit_64:
        #     gecko_path = os.path.join(gecko_dir,"geckodriver_64.exe")
        # else:
        #     gecko_path = os.path.join(gecko_dir,"geckodriver.exe")
        print(gecko_path)
        self.driver = webdriver.Firefox(executable_path=gecko_path, firefox_profile=firefox_profile, firefox_options=options)
        self.driver.set_window_position(0,0)
        self.driver.maximize_window()
        value = 'true-value'
    
        if urllib.request.urlopen(external_url).getcode() == 200:
            #value = 'value'
            self.driver.get(external_url)
        else:
            self.driver.get(internal_url)
            #value = 'true-value'
        if by_formula:
            self.driver.find_element_by_xpath('//*[@'+value+'="exact_formula"]').click()
            inputElement = self.driver.find_element_by_id("identifiers")
            list_formulas = "\n".join(formulas)
            #print(list_formulas)
            inputElement.send_keys(list_formulas)
        else:
            self.driver.find_element_by_xpath('//*[@'+value+'="ms_ready_monoisotopic_mass"]').click()
            self.driver.find_element_by_xpath('//*[@id="mass-select"]/option['+str(ppm)+']').click()
            inputElement = self.driver.find_element_by_id("identifiers")
            list_masses = "\n".join(masses)
            #print(list_masses)
            inputElement.send_keys(list_masses)
        #self.driver.find_element_by_id("display-batch-chemicals-btn").click()
    
        #Select to download the data
        self.driver.find_elements_by_class_name("button")[2].click()
        #self.driver.find_element_by_xpath('//*[@id="batch-search-panel-download"]').click()
    
        # select to download in specific format
        self.driver.find_element_by_xpath('//*[@id="format-select"]/option[@value="tsv"]').click()
    
        #self.driver.find_element_by_id('select-all-properties').click()
        #dtxsid=self.driver.find_element_by_xpath('//*[@value="ATSDRLST"]')
        #self.driver.execute_script("return arguments[0].scrollIntoView();", dtxsid)
        #self.driver.execute_script("window.scrollBy(0, 500)")
        #CASRN
        #self.driver.find_element_by_xpath('//*[@value="casrn" and @class="subinput-checkboxes"]').click()
        #time.sleep(4)
    
        #INChlKey
        self.driver.find_element_by_xpath('//*[@id ="output-headers"]/ul[2]/li[4]/label/input').click()
    
        #IUPAC Name
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[2]/li[5]/label').click()
    
        #Molecular Formula
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[4]/li[1]/label').click()
    
    
        #Average Mass
        #self.driver.find_element_by_xpath('//*[@value="compounds.mol_weight as average_mass"]').click()
    
        # Monoisotopic Mass
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[4]/li[3]/label').click()
    
    
        # OPERA Model Predictions
        #self.driver.find_element_by_xpath('//*[@value="opera_predictions"]').click()
    
        # TEST Model Predictions
        #self.driver.find_element_by_xpath('//*[@id="columns_" and @value="test_predictions"]').click()
    
    
        # Data Sources
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[5]/li[3]/label').click()
    
    
        # Assay Hit Count
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[5]/li[5]/label').click()
    
    
        # EXpoCast
        #self.driver.find_element_by_xpath('//*[@value="expocast"]').click()
        self.driver.find_element_by_xpath('//*[@id="output-headers"]/ul[5]/li[2]/label').click() #expocast option renamed

        #download button
        submit = self.driver.find_elements_by_class_name("button")[3] #.click()
    
        time.sleep(2)
        if submit.is_enabled():
            #print("enabled")
            submit.send_keys(Keys.ENTER)
        return self.driver
    
    
    def close_driver(self):
        self.driver.quit()


#formulas=['C10H14N2O4','C10H5Cl2NO2']
#directory=os.getcwd()
#batch_search(formulas,directory)

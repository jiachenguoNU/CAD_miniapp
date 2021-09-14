# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 22:46:20 2021

@author: Jiachen & Ashiwin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc, accuracy_score, f1_score
from scipy.signal import butter, lfilter,iirnotch,resample_poly
import pywt
import neurokit2 as nk
import pyhrv
import csv
from ecgdetectors import Detectors
import streamlit as st
import joblib
import pandas as pd
from io import BytesIO
from PIL import Image

with st.echo(code_location='below'):
    @st.cache
    #define butterworth filtering function
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    #define band-reject filtering function
    def band_notch(freq, quality, fs):
        nyq = 0.5 * fs
        w0=freq/nyq
        b, a = iirnotch(w0, quality)
        return b, a
    
    
    def notch_filter(data, freq,quality,fs):
        b, a = band_notch(freq, quality, fs)
        y = lfilter(b, a, data)
        return y
    
    #define upsampling function for CAD patients
    def upsamplingCAD(data,freq):
        datalength = len(data)
        num = int(datalength*freq/250)
        y = resample_poly(data,freq,250)
        return y
    st.set_page_config(layout="wide")
    st.title('Automated Coronary Artery Disease Diagnosis')
    st.markdown('by Jiachen Guo and Ashwin Vazhayil for MDS summer course, 2021')
    st.header('1. Import electrocardiogram (ECG)')
    st.markdown('Progress bar-----0%')
    st.progress(0)
    #############################
    #######################
    #################
    #parameters used in filtering
    lowcut = 0.3
    highcut = 15
    reject=50 #powerline inteference noise  
    quality=20 #doesn't require change
    
    
    #%% ecg data import
    uploaded_file = st.file_uploader('Please select a csv file which contains ECG data to upload')
    if uploaded_file is not None:
        reader = pd.read_csv(uploaded_file,names=['Time (s)','Voltage (mv)'])
    #    data = list(reader)
    else:
        st.markdown('Please choose a valid csv file which contains ECG data')
        
    data=np.asarray(reader)
    st.markdown('ECG data has been loaded successfully...')
    st.write(reader)
    
    t=data[:,0]
    s=data[:,1]
    s0=s[1000:-1]
    fs=1/(t[1]-t[0])
    s=butter_bandpass_filter(s, lowcut, highcut, fs, order=1)
    t=t[1000:-1]
    s=s[1000:-1]
    
    #%% derive hrv curve (data collection)
    
    ecg=[] #ecg voltage matrix
    ecgt=[] #ecg time matrix
    rp=[] #index matrix for all r peaks
    rpt=[] #r peak corresponding time
    dur=[] #R peak duration matrix 
    #################
    #######################
    detectors = Detectors(fs)
    r_peaks = detectors.engzee_detector(s)
    rp.append(r_peaks)
    rptime = t[r_peaks]
    rpt.append(rptime)
    #compute duration matrix
    rptime1=np.delete(rptime,-1)    
    rptime2=np.delete(rptime,1)
    dutime=rptime2-rptime1
    dur.append(dutime)
    #store filtered signal in the big matrix ECG for ecg voltage signals    
    ecg.append(s)
    ecgt.append(t)
    
    #visualization for original/filter ECG SIGNALS AND R PEAKS
    st.markdown('_______________________________________________________________________________________________________')
    st.markdown('Want to see the imported ECG data? Toggle 1. plot original and filtered ECG data on the left panel')
    st.markdown('_______________________________________________________________________________________________________')
    if st.sidebar.checkbox('1. Plot original and filtered ECG data'):
        st.markdown('Original and filtered ECG data-----done!!!')
        figecg, (ax1,ax2) = plt.subplots(2,1,figsize=(5,5),gridspec_kw={'height_ratios': [1, 1],'hspace':0.5})
        ax1.plot(t,s0)
        ax1.title.set_text('Original ECG data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        ax2.plot(t,s,'c')
        plt.title('Filtered ECG data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        buf = BytesIO()
        figecg.savefig(buf, format="png")
        st.image(buf)
    
    st.header('2. Derive heart rate variability (HRV)')
    st.markdown('Progress bar-----30%')
    st.progress(30)
    st.markdown('_______________________________________________________________________________________________________')
    st.markdown('Want to see the extracted R-peaks in more detail? Toggle 2. Plot more detailed ECG data with identified R-peaks on the left panel')
    st.markdown('_______________________________________________________________________________________________________')
    if st.sidebar.checkbox('2. Plot more detailed ECG data with identified R-peaks'):
        st.markdown('Filtered ECG data and corresponding R-peaks')
        figecgzoom, axecgzoom = plt.subplots(figsize=(10,4))
        plt.plot(t[1000:3200],s[1000:3200])
        plt.title('ECG data with identified R-peaks (part)')
        plt.xlabel('Time (s)',fontsize=12)
        plt.ylabel('Voltage (mV)',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.plot(t[r_peaks],s[r_peaks],'ro')
        plt.xlim(t[1000],t[3000])
        buf = BytesIO()
        figecgzoom.savefig(buf, format="png")
        st.image(buf)
    
    
    #%% feature extraction
    
    st.markdown('HRV can be obtained through the equation below')
    st.latex(r'''
            HR=\frac{60}{t_{RR}}
            ''')
    timeHealth=[]
    freqHealth=[]
    waveHealth=[] #features for CAD patients
        
    a=dur[0]
    a=a[1:-1] # duration of each different heart beat the 1st element is removed since it's 0
    hrv=60/a #calculate heart rate variability
    
    
    st.markdown('Heart rate variability (HRV) curve is shown in the figure below')
    fighrv, axhrv = plt.subplots(figsize=(6,4))
    plt.plot(np.arange(0,len(hrv)),hrv)
    plt.title('HRV')
    plt.xlabel('Sample')
    plt.ylabel('Heart rate (bpm)')
    buf = BytesIO()
    fighrv.savefig(buf, format="png")
    st.image(buf)
    
    
    st.header('3. Extract features from HRV')
    st.markdown('Progress bar-----60%')
    st.progress(60)
    mean=np.mean(a) #mean of RR interval duration
    sdnn=pyhrv.time_domain.sdnn(nni=a*1000)  #nni is in ms
    sdnn=sdnn['sdnn']/1000 #standard deviation of RR interval duration
    sdsd=pyhrv.time_domain.sdsd(nni=a*1000)  #nni is in ms
    sdsd=sdsd['sdsd']/1000 #standard deviation of RR interval duration differences
    timeHealth=[mean,sdnn,sdsd] # time domain features storation
    freq_all=pyhrv.frequency_domain.welch_psd(nni=a*1000)
    ptotal=freq_all['fft_total']/1000**2 #total power
    pLF=freq_all['fft_norm'][0] #normalized power for low frequency band
    pHF=freq_all['fft_norm'][1] #normalized power for high frequency band https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html#welch-s-method-welch-psd
    ratio=freq_all['fft_ratio'] #LF/HF ratio
    freqplot=freq_all['fft_plot']
    
    st.markdown('_______________________________________________________________________________________________________')
    st.markdown('Want to know how features are extracted from the frequency domain? Toggle 3. Plot power spectral density (PSD) curve on the left panel')
    st.markdown('_______________________________________________________________________________________________________')
    
    if st.sidebar.checkbox('3. Plot power spectral density (PSD) curve'):
        st.markdown('Features from the frequency domain are obtained')
        st.pyplot(freqplot)
    freqHealth=[ptotal,pLF,pHF,ratio] # freq domain features storation
    
    
    
    
    ca3,cd3,cd2,cd1=pywt.wavedec(hrv, 'haar', level=3)
    ya3=pywt.waverec([ca3,np.zeros_like(cd3),np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
    yd3=pywt.waverec([np.zeros_like(ca3),cd3,np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
    yd2=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),cd2,np.zeros_like(cd1)], 'haar')
    yd1=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),np.zeros_like(cd2),cd1], 'haar')
    
    st.markdown('_______________________________________________________________________________________________________')
    st.markdown('Want to know what wavelet decomposition looks like? Toggle 4. Plot wavelet decomposition on the left panel')
    st.markdown('_______________________________________________________________________________________________________')
    
    if st.sidebar.checkbox('4. Plot wavelet decomposition'):
        figwave, axwave = plt.subplots(nrows=4, ncols=1, figsize=(6,4))
        plt.subplot(5,1,1)
        plt.plot(hrv)
        plt.ylabel('HR/bpm')
        plt.title('Wavelet decomposition')
        plt.subplot(5,1,2)
        plt.plot(ya3)
        plt.ylabel('A3')
        plt.subplot(5,1,3)
        plt.plot(yd3)
        plt.ylabel('D3')
        plt.subplot(5,1,4)
        plt.plot(yd2)
        plt.ylabel('D2')
        plt.subplot(5,1,5)
        plt.plot(yd1)
        plt.ylabel('D1')
        plt.xlabel('Sample')
        buf = BytesIO()
        figwave.savefig(buf, format="png")
        st.markdown('Features from the time-frequency domain are obtained through multiresolutional analysis')
        st.image(buf)
        
    waveHealth=[nk.entropy_shannon(ca3),nk.entropy_shannon(cd3)
    ,nk.entropy_shannon(cd2)
    ,nk.entropy_shannon(cd1)
    ,nk.entropy_approximate(ca3)
    ,nk.entropy_approximate(cd3)
    ,nk.entropy_approximate(cd2)
    ,nk.entropy_approximate(cd1)
    ,nk.entropy_sample(ca3)
    ,nk.entropy_sample(cd3)
    ,nk.entropy_sample(cd2)
    ,nk.entropy_sample(cd1)]  
    
    feature=np.hstack((timeHealth,freqHealth,waveHealth))
    feature=np.reshape(feature,(1, -1))
    ftframe=pd.DataFrame(feature,columns=['mean','SDNN','SDSD'
                                        ,'Total power','Low frequency power','High frequency power','LF/HF'
                                        ,'Shannon entropy_CD3','Shannon entropy_CD2','Shannon entropy_CD1','Shannon entropy_CA3'
                                        ,'Approximate entropy_CD3','Approximate entropy_CD2','Approximate entropy_CD1','Approximate entropy_CA3'
                                        ,'Sampling entropy_CD3','Sampling entropy_CD2','Sampling entropy_CD1','Sampling entropy_CA3'])
    st.markdown('Extracted features are summarized below')
    st.write(ftframe)
    #%%classfication
    st.header('4. Classify using pre-trained support vector machine (SVM) model')
    st.markdown('Progress bar-----90%')
    st.progress(90)
    image = Image.open('unsplash.jpg')
    st.image(image,width=500)
    st.markdown('Photo by Robina Weermeijer from Unsplash, free to use under Unsplash License')
    scaler = joblib.load('data_scaler.pkl') 
    X = scaler.transform(feature)
    clf = joblib.load('trainedCADmodel.pkl')
    result=clf.predict(X)
    
    st.markdown("""
    <style>
    .noCAD {
        font-size:25px !important;color:white;font-weight: 700;background-color: lightgreen;border-radius: 0.4rem;
    color: white;
    padding: 0.5rem;
    margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .re {
        font-size:20px !important;font-weight: 700
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .CAD {
        font-size:30px !important;color:white;font-weight: 700;background-color: lightcoral
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="re">Diagnosis result: </p>',unsafe_allow_html=True)
    if result==1:
                st.markdown('<p class="noCAD">Coronary Artery Disease is not detected! </p>',unsafe_allow_html=True)
    else:
        st.markdown('<p class="CAD">Coronary Artery Disease is detected </p> ',unsafe_allow_html=True)
    st.header('5. Code')
    st.markdown('Progress bar-----100%')
    st.progress(100)
    st.markdown('The code of this program is shown below:')

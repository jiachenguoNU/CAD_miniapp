# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 22:46:20 2021

@author: Jiachen & Ashwin
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
st.header('Import electrocardiogram (ECG)')
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
st.markdown('Download sample file 1 for healthy people: https://drive.google.com/uc?export=download&id=18ZU5soT9sNLJg424ROomB1ivtzO7_216 ')
st.markdown('Download sample file 2 for healthy people: https://drive.google.com/uc?export=download&id=1R8uy8pbgLBKNb3vfWZFMhdsKA4T3AHcG ')
st.markdown('Download sample file 3 for healthy people: https://drive.google.com/uc?export=download&id=1cG95GGSrZ1EjoxLmehWyJfhkhOTfWbNV ')
st.markdown('Download sample file 1 for CAD patient: https://drive.google.com/uc?export=download&id=1dYvOE-TUp7pnuMABis4u-8_nr8KC_h-L')
st.markdown('Download sample file 2 for CAD patient: https://drive.google.com/uc?export=download&id=1vDrn7pyexfAaC4Pmu3P6ZbcI3Hed1akI')
st.markdown('Download sample file 3 for CAD patient: https://drive.google.com/uc?export=download&id=14C8JdHps-xo7N5Br09lhif0_XFNHAOc6')
#%% ecg data import
uploaded_file = st.file_uploader('To begin, please select a csv file which contains ECG data to upload.')
if uploaded_file is not None:
    reader = pd.read_csv(uploaded_file,names=['Time (s)','Voltage (mv)'])
#    data = list(reader)
else:
    st.markdown('Please choose a valid csv file which contains ECG data.')
    
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
st.header('A. Data preparation')
st.markdown('Progress bar-----10%')
st.progress(10)
st.subheader('A.1 ECG visualization')
st.markdown('_______________________________________________________________________________________________________')
st.markdown('Want to see the imported ECG? Check A.1 on the left panel to plot the ECG signal.')
st.markdown('_______________________________________________________________________________________________________')
if st.sidebar.checkbox('A.1 ECG visualization'):
    st.markdown('Original and filtered ECG data-----done!!!')
    figecg, (ax1,ax2) = plt.subplots(2,1,figsize=(10,5),gridspec_kw={'height_ratios': [1, 1],'hspace':0.5})
    ax1.plot(t,s0)
    ax1.title.set_text('Original ECG data (all)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    ax2.plot(t[1000:2010],s0[1000:2010],'c')
    plt.title('Original ECG data (part)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    buf = BytesIO()
    figecg.savefig(buf, format="png")
    st.image(buf)

    st.subheader('A.2 ECG denoising')
    st.markdown('_______________________________________________________________________________________________________')
    st.markdown('''The original ECG data consists some low-frequency and high-frequency noise, which may affect the accuracy of the R peak detection algorithm. Therefore,
                ECG signal has to be denoised. To this end, the Butterworth filter is used to eliminate the noise.''')
    st.markdown('Check A.2 to start denoising the ECG data.')            
    st.markdown('_______________________________________________________________________________________________________')
    
    if st.sidebar.checkbox('A.2 ECG data denoising'):
        st.markdown('ECG denoising-----done!!!')
        figecg, (ax1,ax2) = plt.subplots(2,1,figsize=(10,5),gridspec_kw={'height_ratios': [1, 1],'hspace':0.5})
        ax1.plot(t,s0)
        ax1.title.set_text('Filtered ECG data (all)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        ax2.plot(t[1000:2010],s[1000:2010],'c')
        plt.title('Filtered ECG data (part)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        buf = BytesIO()
        figecg.savefig(buf, format="png")
        st.image(buf)
        
        st.subheader('A.3 R peaks detection')
        st.markdown('_______________________________________________________________________________________________________')
        st.markdown('''After denoising, R peaks can be automatically identified using the R peak detection algorithm.''')
        st.markdown('Check A.3 to detect R peaks of the ECG signal.')
        st.markdown('_______________________________________________________________________________________________________')
        if st.sidebar.checkbox('A.3 R peaks detection'):
            st.markdown('R peaks detection-----done!!!')
            figecgzoom, axecgzoom = plt.subplots(figsize=(10,4))
            plt.plot(t[1000:3200],s[1000:3200])
            plt.title('ECG data with identified R-peaks (part)')
            plt.xlabel('Time (s)',fontsize=12)
            plt.ylabel('Voltage (mV)',fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.plot(t[r_peaks],s[r_peaks],'ro')
            plt.xlim(t[1500],t[3000])
            buf = BytesIO()
            figecgzoom.savefig(buf, format="png")
            st.image(buf)
            
            
            st.subheader('A.4 HRV derivation')
            st.markdown('_______________________________________________________________________________________________________')
            st.markdown('''Upon obtaining R peaks, duration of each different heart beat can be obtained. Heart rate variability (HRV) can then be derived. HRV measures
                        the variation of heart beats from beat to beat. This is significant as patients who exhibit CAD often have a reduced rhythm of HRV and can be compared to 
                        healthy patients.''')
            st.markdown('HRV can be obtained through the equation below:')
            st.latex(r'''
                        HR=\frac{60}{t_{RR}}
                        ''')
            st.markdown('Check A.4 to derive the HRV curve.')
            st.markdown('_______________________________________________________________________________________________________')
                                    
            if st.sidebar.checkbox('A.4 HRV derivation'):
                st.markdown('HRV derivation-----done!!!')
            
                
                timeHealth=[]
                freqHealth=[]
                waveHealth=[] #features for CAD patients
                    
                a=dur[0]
                a=a[1:-1] # duration of each different heart beat the 1st element is removed since it's 0
                hrv=60/a #calculate heart rate variability
                
            
                fighrv, axhrv = plt.subplots(figsize=(10,4))
                plt.plot(np.arange(0,len(hrv)),hrv)
                plt.title('HRV')
                plt.xlabel('Sample')
                plt.ylabel('Heart rate (bpm)')
                buf = BytesIO()
                fighrv.savefig(buf, format="png")
                st.image(buf)

                
                
                #%% feature extraction
                st.header('B. Feature extraction')
                st.markdown('Progress bar-----40%')
                st.progress(40)
                
                st.markdown('''The timing of the heartbeat is controlled by the heart’s electrical system. As a result, if abnormality
                            is found in the cardiac electrical system, the heart rhyme might be abnormal and the heart may not function 
                            properly. Therefore, HRV can be used to diagnose the heart’s functionality.''')
                
                st.subheader('B.1 Time domain features')
                st.markdown('_______________________________________________________________________________________________________')
                st.markdown('''To fully uncover the hidden characteristics behind HRV, features from the time domain, frequency domain and time-
                            frequency domain can be extracted.''')
                st.markdown('''Check B.1 to extract features from the time domain.''')
                st.markdown('_______________________________________________________________________________________________________')
                if st.sidebar.checkbox('B.1 Time domain features'):

                    
                    mean=np.mean(a) #mean of RR interval duration
                    sdnn=pyhrv.time_domain.sdnn(nni=a*1000)  #nni is in ms
                    sdnn=sdnn['sdnn']/1000 #standard deviation of RR interval duration
                    sdsd=pyhrv.time_domain.sdsd(nni=a*1000)  #nni is in ms
                    sdsd=sdsd['sdsd']/1000 #standard deviation of RR interval duration differences
                    timeHealth=[mean,sdnn,sdsd] # time domain features storation
                    st.markdown('''Common linear statistical values for a time series include the mean, standard deviation, and standard
                                deviation of successive differences. These values can be used effectively to assess the overall behavior
                                of the signal in the time domain.  As a result, three linear features are extracted in the time domain.''')
                    st.markdown('The mean heartbeat duration is defined below:')
                    st.latex(r'''
                        \bar{\Delta}_{t}=\sum_{j=1}^{n} \Delta_{t_{i}} 
                        ''')
                    st.markdown('The standard deviation of heartbeat duration (SD) is defined as follows:')
                    st.latex(r'''
                        S D=\sqrt{\frac{1}{n-1} \sum_{j=1}^{n}\left(\Delta_{t_{j}}-\overline{\Delta}_t\right)^{2}}  
                        ''')
                    st.markdown('The standard deviation of heartbeat duration differences (SDSD) is defined in the equation below:')
                    st.latex(r'''
                         S D S D=\sqrt{\frac{1}{n-1} \sum_{j=1}^{n}\left(\Delta\left(\Delta_{t_{j}}\right)-\overline{\Delta\left(\Delta_{t}\right)}\right)^{2}}  
                        ''')                            
                    featuret=np.array([mean,sdnn,sdsd])
                    featuret=np.hstack(featuret)
                    featuret=np.reshape(featuret,(1,-1))
                    ftframe=pd.DataFrame(featuret,columns=['mean','SD','SDSD'])
                    st.markdown('Features extracted from the time domain are summarized below:')
                    st.write(ftframe)
                    
                    
                    
                    st.subheader('B.2 Frequency domain features')
                    st.markdown('_______________________________________________________________________________________________________')
                    st.markdown('''The power distribution of HRV in the frequency domain can effectively
                                reflect the functionality of the cardiac autonomic modulation. For example, it has been shown that total power can reflect abnormal
                                autonomic activity; the power
                                in the high frequency (HF) domain can be used to represent parasympathetic modulation, whereas the power in the 
                                low-frequency domain corresponds to the sympathetic modulation. The ratio between LF
                                and HF can also be deemed as an indicator for the balance between sympathetic and parasympathetic modulation. 
                                Therefore, in order to extract features in the frequency domain, the power spectral density (PSD) of HRV is calculated first.''')
                    
                    st.markdown('Check B.2 to derive the PSD curve of HRV.')
                    st.markdown('_______________________________________________________________________________________________________')
                    if st.sidebar.checkbox('B.2 Frequency domain features'):
                    
                        freq_all=pyhrv.frequency_domain.welch_psd(nni=a*1000)
                        ptotal=freq_all['fft_total']/1000**2 #total power
                        pLF=freq_all['fft_norm'][0] #normalized power for low frequency band
                        pHF=freq_all['fft_norm'][1] #normalized power for high frequency band https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html#welch-s-method-welch-psd
                        ratio=freq_all['fft_ratio'] #LF/HF ratio
                        freqplot=freq_all['fft_plot']
                        
                        

                        st.pyplot(freqplot)
                        freqHealth=[ptotal,pLF,pHF,ratio] # freq domain features storation
                        st.markdown('''Note that there are 3 regions of different colors in the figure above. The red one represents power distribution in the
                                    low-frequency domain whereas the blue one represents power distribution in the high-frequency domain. The ratio of the
                                    low-frequency power and high-frequency power LF/HF can also be calculated.''')

                        featuref=np.array([ptotal,pLF,pHF,ratio])
                        featuref=np.hstack(featuref)
                        featuref=np.reshape(featuref,(1,-1))
                        ftframe=pd.DataFrame(featuref,columns=['Total power','Low-frequency power','High-frequency power','LF/HF'])
                        st.markdown('Features extracted from the frequency domain are summarized below:')
                        st.write(ftframe)
                        
                        st.subheader('B.3 Time-frequency domain features')
                        st.markdown('_______________________________________________________________________________________________________')
                        st.markdown('''
                                    HRV is highly nonlinear and non-stationary. As a consequence, features from the time domain and frequency domain might 
                                    not be sufficient to reflect the hidden complexities of HRV. Accordingly, time-frequency transformation is also used to 
                                    extract more features. Here discrete wavelet transform is used to decompose HRV.
                                    ''')
                        
                        st.markdown('Check B.3 to carry out wavelet decomposition.')
                        st.markdown('_______________________________________________________________________________________________________')
                        
                        if st.sidebar.checkbox('B.3 Time-frequency domain features'):
                        
                            ca3,cd3,cd2,cd1=pywt.wavedec(hrv, 'haar', level=3)
                            ya3=pywt.waverec([ca3,np.zeros_like(cd3),np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
                            yd3=pywt.waverec([np.zeros_like(ca3),cd3,np.zeros_like(cd2),np.zeros_like(cd1)], 'haar')
                            yd2=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),cd2,np.zeros_like(cd1)], 'haar')
                            yd1=pywt.waverec([np.zeros_like(ca3),np.zeros_like(cd3),np.zeros_like(cd2),cd1], 'haar')
                            
                            
                            
                            
                            

                            figwave, axwave = plt.subplots(nrows=4, ncols=1, figsize=(10,4))
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
                            st.markdown('Wavelet decomposition-----done!!!')
                            st.image(buf)
                            st.markdown('''Nonlinear feature extraction techniques are used for wavelet coefficients at each different level. 3 kinds of entropies, namely
                                        Shannon entropy, approximation entropy, and sampling entropy are calculated. Among them, Shannon entropy measures the data uncertainty
                                        and variability; approximation entropy quantifies the amount of regularity and the unpredictability of fluctuations; sampling entropy is
                                        generally used to assess complexities of physiological signals.

                                        
                                        ''')
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
                            
                            featurew=np.hstack(waveHealth)
                            featurew=np.reshape(featurew,(1, -1))
                            fwframe=pd.DataFrame(featurew,columns=['Shannon entropy_CD3','Shannon entropy_CD2','Shannon entropy_CD1','Shannon entropy_CA3'
                                                                ,'Approximate entropy_CD3','Approximate entropy_CD2','Approximate entropy_CD1','Approximate entropy_CA3'
                                                                ,'Sampling entropy_CD3','Sampling entropy_CD2','Sampling entropy_CD1','Sampling entropy_CA3'])
                            st.markdown('Features extracted from the time-frequency domain are summarized below:')
                            st.write(fwframe)
                            
                            

                            feature=np.hstack((timeHealth,freqHealth,waveHealth))
                            feature=np.reshape(feature,(1, -1))
                            ftframe=pd.DataFrame(feature,columns=['mean','SD','SDSD'
                                                                ,'Total power','Low-frequency power','High-frequency power','LF/HF'
                                                                ,'Shannon entropy_CD3','Shannon entropy_CD2','Shannon entropy_CD1','Shannon entropy_CA3'
                                                                ,'Approximate entropy_CD3','Approximate entropy_CD2','Approximate entropy_CD1','Approximate entropy_CA3'
                                                                ,'Sampling entropy_CD3','Sampling entropy_CD2','Sampling entropy_CD1','Sampling entropy_CA3'])
                            st.markdown('_______________________________________________________________________________________________________')
                            st.markdown('''To conclude, 19 features have been extracted from HRV: namely 3 features from the time domain, 4 features from the frequency domain and 12 features from
                                        the time-frequency domain.''')
                            st.markdown('All extracted features are summarized below:')
                            st.markdown('_______________________________________________________________________________________________________')
                            st.write(ftframe)

                            #%%classfication
                            st.header('C. Classification using pre-trained support vector machine (SVM) model')
                            st.markdown('Progress bar-----70%')
                            st.progress(70)
                            st.markdown('_______________________________________________________________________________________________________')
                            st.markdown('''With the obtained input features, support vector machine (SVM), one of the supervised learning 
                                        algorithms, is used to uncover the hidden relationship between the input features and the CAD 
                                        diagnosis result. During the 2-class classification process, SVM strives to find a decision boundary 
                                        (hyperplane) to divide the input feature space into two parts.''')
                            st.markdown('''The unique aspect of SVM is that it 
                                        ensures the maximized distance between the decision boundary and data points. The nearest data points 
                                        to the decision boundary are named as support vectors. Meanwhile, SVM has been shown to be very 
                                        effective for problems with relatively small datasets.''')
                            st.markdown('Check C.1 to see the diagnosis result.')
                            st.markdown('_______________________________________________________________________________________________________')
                            image = Image.open('unsplash.jpg')
                            st.image(image,width=500)
                            st.markdown('Photo by Robina Weermeijer from Unsplash, free to use under Unsplash License')
                            scaler = joblib.load('data_scaler.pkl') 
                            X = scaler.transform(feature)
                            clf = joblib.load('trainedCADmodel.pkl')
                            result=clf.predict(X)
                            if st.sidebar.checkbox('C.1 Classification using SVM'):
                            
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

                                st.header('D. Code of the mini-app')
                                st.markdown('Progress bar-----90%')
                                st.progress(90)
                                st.markdown('If you want to see the code of this mini-app, please check D.1.')
                                if st.sidebar.checkbox('D.1 Code'):
                                    st.markdown('The code of this program can be found on: https://github.com/jiachenguoNU/CAD_miniapp')


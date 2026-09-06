import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, json, time
from scipy.signal import fftconvolve, butter, sosfilt
from aaa import signals, analysis
sr=48000
def speaker_ir(dist, polarity=1.0, tweeter_pol=1.0, mode_hz=45.0, seed=0):
    rng=np.random.default_rng(seed)
    n=int(0.8*sr); ir=np.zeros(n)
    d=int(dist/343*sr)
    # woofer: lowpassed click ; tweeter: highpassed click (crossover ~2.5k)
    imp=np.zeros(n); imp[d]=1.0
    sos_lo=butter(2,2500/(sr/2),'low',output='sos'); sos_hi=butter(2,2500/(sr/2),'high',output='sos')
    woof=sosfilt(sos_lo,imp); tw=sosfilt(sos_hi,imp)
    ir+=polarity*(woof+tweeter_pol*tw)
    # reflections
    for k in range(12):
        t=d+int(rng.uniform(0.004,0.06)*sr); ir[t]+=polarity*rng.uniform(-0.3,0.3)
    # room mode: decaying sinusoid at mode_hz
    t=np.arange(n)/sr
    ir+=polarity*0.6*np.sin(2*np.pi*mode_hz*t)*np.exp(-t/0.4)*(t>d/sr)
    # tail
    ir+=polarity*rng.standard_normal(n)*0.02*np.exp(-t/0.3)
    return ir
sig,markers,inv=signals.build_measurement(2,sr)
latency=int(0.137*sr)
for case,(polR,twR) in {"good":(1,1),"R reversed":(-1,1),"R tweeter reversed":(1,-1)}.items():
    irL=speaker_ir(4.0); irR=speaker_ir(4.3,polR,twR,seed=1)
    rec=fftconvolve(sig[:,0],irL)+fftconvolve(sig[:,1],irR)
    rec=np.concatenate([np.zeros(latency),rec]); rec+=np.random.default_rng(5).standard_normal(len(rec))*0.002
    t0=time.time()
    irs,lat,onsets,full=analysis.extract_irs(rec,markers,inv,sr)
    pe=analysis.polarity_evidence(irs,onsets,rec,markers,lat,sr,distances=[4.0,4.3])
    rm=analysis.room_mode_evidence(irs,sr,dims=[3.8,4.5,2.4])
    print(f"== {case} ({time.time()-t0:.1f}s) latency {lat} (true {latency}) onsets {onsets}")
    print(" verdict:",pe["algorithmic_verdict"])
    print(" pair:",pe["pairs"]); print(" burst:",{k:v for k,v in pe["burst_test"].items() if 'runs' not in k})
    print(" chans:",{k:{kk:vv for kk,vv in v.items()} for k,v in pe["channels"].items()})
    print(" peaks:",rm["peaks"][:3]); print(" dips:",rm["dips"])

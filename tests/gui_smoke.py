import sys, os, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from scipy.signal import fftconvolve
from aaa import app as A, audio_io, session
from sim_test import speaker_ir
# fake audio
def fake_play_and_record(sig, sink, source, workdir, sr=48000, tail_s=1.5, ac=None, af=None, progress=None, manual_wait_s=None):
    irL=speaker_ir(4.0); irR=speaker_ir(4.3,-1,1,seed=1)
    rec=fftconvolve(sig[:,0],irL)+fftconvolve(sig[:,1],irR)
    rec=np.concatenate([np.zeros(6000),rec])+np.random.default_rng(5).standard_normal(len(rec)+6000)*0.002
    os.makedirs(workdir,exist_ok=True); return rec, rec[:,None]
audio_io.play_and_record = fake_play_and_record
audio_io.record_only = lambda *a, **k: np.random.default_rng(1).standard_normal(48000)*1e-4
import tempfile
TMP = tempfile.mkdtemp(prefix='aaa_test_')
session.CONFIG = os.path.join(TMP, 'config.json')
session.ROOT = os.path.join(TMP, 'sessions')
A.load_config = lambda: {}
app = A.App()
app.update()
app.vars['room_dims'].set('3.8 4.5 2.4'); app.vars['dist_l'].set('4.0'); app.vars['dist_r'].set('4.3'); app.vars['setup'].set('test amp')
# pick a pw source to avoid alsa probe
for i,s in enumerate(app.sources):
    if s['kind']=='pw': app.src_cb.current(i); break
app.save_setup(); app.update()
app.set_mic_profile([[20,-15],[100,-3],[1000,0],[10000,2],[20000,-5]], 'test'); app.update()
app.show_step(2); app.update(); app.noise_floor()
def wait():
    t=time.time()
    while app.busy and time.time()-t<60: app.update(); time.sleep(0.05)
wait(); app.level_test(); wait()
print('level text:', app.level_text.get('1.0','end').strip())
app.show_step(3); app.update(); app.measure(); wait(); app.update()
assert app.last, 'no measurement'
print('verdict', app.last['ev']['polarity']['algorithmic_verdict'])
print('peaks', [(p['freq'],p['excess_db']) for p in app.last['ev']['room_modes']['peaks']])
for i in (4,5,6): app.show_step(i); app.update()
print('pol text head:', app.pol_text.get('1.0','4.0').strip())
print('export head:', app.exp_text.get('1.0','5.0').strip())
print('pngs', [os.path.exists(p) for p in app.last['pngs'].values()])
app.destroy(); print('GUI OK')

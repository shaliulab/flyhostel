import numpy as np, re

# --- your SLEAP node_names -> roles ---
# your labels: f/m/r = fore/mid/rear, L/R = fly-left/right, L = leg tip, J = joint(knee)
# proboscis + fore/rear knees (fLLJ,fRLJ,rLLJ,rRLJ) exist in your data but the
# APF default config ignores them, so they're intentionally not mapped.
N = dict(head='head', thorax='thorax', abdomen='abdomen',
         wing_l='lW',  wing_r='rW',
         tip_fl='fLL', tip_ml='mLL', tip_bl='rLL',
         tip_fr='fRL', tip_mr='mRL', tip_br='rRL',
         knee_ml='mLLJ', knee_mr='mRLJ')
LEFT_NODES  = ['wing_l','tip_fl','tip_ml','tip_bl','knee_ml']
RIGHT_NODES = ['wing_r','tip_fr','tip_mr','tip_br','knee_mr']


# --- CALIBRATION: measure these on YOUR rig. Do NOT reuse Branson's numbers. ---
FLIP_Y = True

# synthetic offsets as fractions of body length |head-abdomen|; tune if reconstruction looks off
F_THLEN, F_THWID, F_HDWID, F_EYEFWD, F_FBWID = 0.28, 0.16, 0.14, 0.06, 0.22

APF_KEYPOINTS = ['wing_left','wing_right','antennae_midpoint','right_eye','left_eye',
    'left_front_thorax','right_front_thorax','base_thorax','tip_abdomen',
    'right_middle_femur_base','right_middle_femur_tibia_joint','left_middle_femur_base',
    'left_middle_femur_tibia_joint','right_front_leg_tip','right_middle_leg_tip',
    'right_back_leg_tip','left_back_leg_tip','left_middle_leg_tip','left_front_leg_tip']
APF = {k:i for i,k in enumerate(APF_KEYPOINTS)}

def _unit(v):                                    # v: (2,T)
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return v / np.where(n==0, np.nan, n)

def synthesize(nd):                              # nd: role -> (2,T) in mm, centered, y-up
    H, Tx, A = nd['head'], nd['thorax'], nd['abdomen']; T = H.shape[1]
    u = _unit(H - A)                             # forward, per frame
    Lc = np.nanmean(np.stack([nd[k] for k in LEFT_NODES ]), axis=0)
    Rc = np.nanmean(np.stack([nd[k] for k in RIGHT_NODES]), axis=0)
    lat = Lc - Rc
    nL = _unit(lat - np.sum(lat*u,0,keepdims=True)*u)          # data-driven fly-left
    rot = np.stack([-u[1], u[0]])                             # +90 fallback
    sgn = np.sign(np.nansum(rot*lat))
    bad = ~np.isfinite(nL).all(0); nL[:,bad] = (sgn*rot)[:,bad]

    body = np.linalg.norm(np.nanmedian(H - A, axis=1))
    Lth,Wth,Whd,Efw,Wfb = (f*body for f in (F_THLEN,F_THWID,F_HDWID,F_EYEFWD,F_FBWID))
    bthx, fthx = Tx, Tx + Lth*u
    midthx, mideye = 0.5*(bthx+fthx), fthx + Efw*u

    X = np.full((19,2,T), np.nan, np.float32); put = lambda k,p: X.__setitem__(APF[k], p)
    put('base_thorax',bthx); put('tip_abdomen',A); put('antennae_midpoint',H)
    put('wing_left',nd['wing_l']); put('wing_right',nd['wing_r'])
    put('left_front_leg_tip',nd['tip_fl']);  put('right_front_leg_tip',nd['tip_fr'])
    put('left_middle_leg_tip',nd['tip_ml']); put('right_middle_leg_tip',nd['tip_mr'])
    put('left_back_leg_tip',nd['tip_bl']);   put('right_back_leg_tip',nd['tip_br'])
    put('left_middle_femur_tibia_joint',nd['knee_ml'])
    put('right_middle_femur_tibia_joint',nd['knee_mr'])
    put('left_front_thorax',  fthx - (Wth/2)*nL); put('right_front_thorax', fthx + (Wth/2)*nL)
    put('left_eye',  mideye - (Whd/2)*nL);        put('right_eye',  mideye + (Whd/2)*nL)
    put('left_middle_femur_base',  midthx + (Wfb/2)*nL)
    put('right_middle_femur_base', midthx - (Wfb/2)*nL)
    return X

def group_key(p):                                 # ADAPT to your path scheme
    return re.search(r'(FlyHostel\d+/\d+/\d{8}_\d{6})', p).group(1)

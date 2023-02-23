import sacred
from sacred.observers import FileStorageObserver
from simulation_logic import do_one_parameter_config
import numpy as np

ex = sacred.Experiment("simulation")
ex.observers.append(FileStorageObserver("data"))


@ex.config
def cfg():    
    rx1 = 0.75
    ry1 = 0.125
    dist = 0.15
    cx1 = -rx1 - dist /2.
    cy1 = 0
    AS1 = 15.
    cx2 = -cx1
    cy2 = cy1
    rx2 = rx1
    ry2 = ry1
    AS2 = AS1
    nr = 1
    ang = nr / 180. * np.pi
    nft = 80
    nx1 = 31
    ny1 = 81
    nx2 = 100
    ny2 = 50
    q = 1.035
    err_0 = 1e-4
    err_w = 1e-4
    num_iter = 500
    dt0 = 5e-4
    aq = 0
    xis = 5e-2
    t_save = 0
    n_expos = 80
    ns = 5
    dt_save = ns * n_expos * dt0
    gpx = 10
    gpy = 8
    gamma = 0.045
    zeta = 15.
    
    kkappa = 0
    lambdat = 0
    gp = 0
    
    sp = 1.5
    
    bf = 10 * gamma
    
    psi = 0

    poff = 20.
        
    pn1 = 15.
    pn2 = pn1 / 20.
    
    pu = poff / 2.

    
    pm = 12.
    bell_shape_x = 8.
    bell_shape_y = 8.

    eta = 1.
    a = 1.
    ka = 0.8
    pei0 = 0.001



@ex.automain
def run_one_simulation(_config, _run):
    do_one_parameter_config(
        cx1=_config["cx1"],
        cy1=_config["cy1"],
        rx1=_config["rx1"],
        ry1=_config["ry1"],
        AS1=_config["AS1"],
        cx2=_config["cx2"],
        cy2=_config["cy2"],
        rx2=_config["rx2"],
        ry2=_config["ry2"],
        AS2=_config["AS2"],
        ang=_config["ang"],
        dist=_config["dist"],
        nx1=_config["nx1"],
        ny1=_config["ny1"],
        nx2=_config["nx2"],
        ny2=_config["ny2"],
        q=_config["q"],
        err_0=_config["err_0"],
        err_w=_config["err_w"],
        num_iter=_config["num_iter"],
        dt0=_config["dt0"],
        aq=_config["aq"],
        t_save=_config["t_save"],
        dt_save=_config["dt_save"],
        gpx=_config["gpx"],
        gpy=_config["gpy"],
        gamma=_config["gamma"],
        zeta=_config["zeta"],
        kkappa=_config["kkappa"],
        lambdat=_config["lambdat"],
        gp=_config["gp"],
        poff=_config["poff"],
        pn1=_config["pn1"],
        pn2 =_config["pn2"],
        pu=_config["pu"],
        pm=_config["pm"],
        bell_shape_x=_config["bell_shape_x"],
        bell_shape_y=_config["bell_shape_y"],
        xis=_config["xis"],
        eta=_config["eta"],
        n_expos=_config["n_expos"],
        sp=_config["sp"],
        a = _config["a"],
        bf= _config["bf"],
        ka= _config["ka"],
        psi= _config["psi"],
        pei0= _config["pei0"],
        nft= _config["nft"],
        SAVE=True,
        ex=ex,
        SAVE_data=True
    )

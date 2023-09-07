import pytest
import numpy as np
from scipy.integrate import odeint
import WallSpeed
from .TestModel import TestModelTemplate


#These tests are all based on a comparison between the classes HydroTemplateModel and Hydro used with TestTemplateModel
N = 100
rng = np.random.default_rng(1)

def test_JouguetVelocity():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallSpeed.Hydro(model)
        hydroTemplate = WallSpeed.HydroTemplateModel(model)
        res1[i] = hydro.findJouguetVelocity()
        res2[i] = hydroTemplate.findJouguetVelocity()
    np.testing.assert_allclose(res1,res2,rtol = 10**-6,atol = 0)

def test_findMatching():
    res1,res2 = np.zeros((N,4)),np.zeros((N,4))
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    vw = rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallSpeed.Hydro(model,1e-10,1e-10)
        hydroTemplate = WallSpeed.HydroTemplateModel(model,1e-10,1e-10)
        res1[i] = hydro.findMatching(vw[i])
        res2[i] = hydroTemplate.findMatching(vw[i])
        if np.isnan(res1[i,0]):
            res1[i] = [0,0,0,0]
        if np.isnan(res2[i,0]):
            res2[i] = [0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-2,atol = 0)

def test_findvwLTE():
    res1,res2 = np.zeros(N),np.zeros(N)
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallSpeed.Hydro(model)
        hydroTemplate = WallSpeed.HydroTemplateModel(model)
        res1[i] = hydro.findvwLTE()
        res2[i] = hydroTemplate.findvwLTE()
    np.testing.assert_allclose(res1,res2,rtol = 10**-4,atol = 0)

def test_findHydroBoundaries():
    res1,res2 = np.zeros((N,4)),np.zeros((N,4))
    psiN = 1-0.5*rng.random(N)
    alN = (1-psiN)/3+rng.random(N)
    cs2 = 1/4+(1/3-1/4)*rng.random(N)
    cb2 = cs2-(1/3-1/4)*rng.random(N)
    vw = rng.random(N)
    for i in range(N):
        model = TestModelTemplate(alN[i],psiN[i],cb2[i],cs2[i],1,1)
        hydro = WallSpeed.Hydro(model,1e-10,1e-10)
        hydroTemplate = WallSpeed.HydroTemplateModel(model,1e-10,1e-10)
        res1[i] = hydro.findHydroBoundaries(vw[i])[:4] # [:4] hack!
        res2[i] = hydroTemplate.findHydroBoundaries(vw[i])
        if np.isnan(res1[i,0]):
            res1[i] = [0,0,0,0]
        if np.isnan(res2[i,0]):
            res2[i] = [0,0,0,0]
    np.testing.assert_allclose(res1,res2,rtol = 10**-3,atol = 0)

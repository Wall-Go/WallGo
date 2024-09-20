/***************************************************************************
 * WallGo/Collisions example file -- SMQCD.cpp.
 * Calculates collision integrals for [TODO what theory?] theory.
 * This example is short on purpose.
 * See QCD.cpp for a more verbose and documented example.
****************************************************************************/

#include <filesystem>

#include "WallGo/WallGoCollision.h"


void defineParticlesSMQCD(wallgo::ModelDefinition& inOutModelDef)
{

    wallgo::ParticleDescription higgs;
    higgs.name = "H";
    higgs.index = 0;
    higgs.type = wallgo::EParticleType::eBoson;
    higgs.bInEquilibrium = true;
    higgs.bUltrarelativistic = true;
    inOutModelDef.defineParticleSpecies(higgs);

    wallgo::ParticleDescription top;
    top.name = "t";
    top.index = 1;
    top.type = wallgo::EParticleType::eFermion;
    top.bInEquilibrium = false;
    top.bUltrarelativistic = true;
    inOutModelDef.defineParticleSpecies(top);

    wallgo::ParticleDescription topBar = top;
    topBar.name = "tbar";
    topBar.index = 2;
    inOutModelDef.defineParticleSpecies(topBar);

    wallgo::ParticleDescription bottom;
    bottom.name = "b";
    bottom.index = 3;
    bottom.type = wallgo::EParticleType::eFermion;
    bottom.bInEquilibrium = true;
    bottom.bUltrarelativistic = true;
    inOutModelDef.defineParticleSpecies(bottom);

    wallgo::ParticleDescription bottomBar = bottom;
    bottomBar.name = "bbar";
    bottomBar.index = 4;
    inOutModelDef.defineParticleSpecies(bottomBar);

    wallgo::ParticleDescription zBoson;
    zBoson.name = "Z";
    zBoson.index = 5;
    zBoson.type = wallgo::EParticleType::eBoson;
    zBoson.bInEquilibrium = true;
    zBoson.bUltrarelativistic = true;
    inOutModelDef.defineParticleSpecies(zBoson);

    wallgo::ParticleDescription wBoson = zBoson;
    wBoson.name = "W";
    wBoson.index = 6;
    inOutModelDef.defineParticleSpecies(wBoson);

    wallgo::ParticleDescription wBarBoson = wBoson;
    wBarBoson.name = "Wbar";
    wBarBoson.index = 7;
    inOutModelDef.defineParticleSpecies(wBarBoson);

    wallgo::ParticleDescription gluon;
    gluon.name = "g";
    gluon.index = 8;
    gluon.type = wallgo::EParticleType::eBoson;
    gluon.bInEquilibrium = true;
    gluon.bUltrarelativistic = true;
    inOutModelDef.defineParticleSpecies(gluon);

}

int main() 
{
	std::cout << "### Running WallGo collision example : SMQCD ###" << std::endl;

    wallgo::initializeRNG();

    //~ Begin model def
    wallgo::ModelDefinition modelDef;

    wallgo::ModelParameters params;
    params.addOrModifyParameter("lam", 0.15);
    params.addOrModifyParameter("yt", 1.0);
    params.addOrModifyParameter("gW", 0.6);
    params.addOrModifyParameter("gs", 1.3);

    modelDef.defineParameters(params);
    
    defineParticlesSMQCD(modelDef);
    //~ End model def

    wallgo::PhysicsModel model(modelDef);
    model.loadMatrixElements("MatrixElements/SMQCD.json", true);
	
	const int basisSizeN = 3;
    wallgo::CollisionTensor collisionTensor = model.createCollisionTensor(basisSizeN);


    wallgo::CollisionTensorVerbosity verbosity;
    verbosity.bPrintElapsedTime = true;
    verbosity.progressReportPercentage = 0.50;
    verbosity.bPrintEveryElement = true;

    // TEST with very short monte carlo. Currently they blow up because matrix elements don't have IR cutoffs
    wallgo::IntegrationOptions integrationOptions;
    integrationOptions.calls = 1000;
    integrationOptions.maxTries = 1;
    integrationOptions.maxIntegrationMomentum = 20;

    std::cout << "## Begin collision integrations ##" << std::endl;
    wallgo::CollisionTensorResult result = collisionTensor.computeIntegralsAll(integrationOptions, verbosity);

    return 0;
}
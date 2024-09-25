/***************************************************************************
 * WallGoCollision example file -- SM.cpp.
 * Calculates collision integrals for the Standard Model,
 * assuming out-of-equilibrium top/gluon/W boson.
 * This example is short on purpose;
 * see QCD.cpp for a more verbose and documented example.
****************************************************************************/

#include <filesystem>

#include "WallGo/WallGoCollision.h"


void defineParticlesSM(wallgo::ModelDefinition& inOutModelDef)
{
    using wallgo::ParticleDescription;
    using wallgo::EParticleType;

    std::vector<ParticleDescription> particles;

    // Make everything ultrarelativistic. Last argument is flag for in-equilibrium.
    // We take top/gluon/W to be out-of-eq, although the example runs even if everything is allowed to deviate from equilibrium.

    inOutModelDef.defineParticleSpecies(ParticleDescription("TopL"         ,  0, EParticleType::eFermion, false));
    inOutModelDef.defineParticleSpecies(ParticleDescription("TopR"         ,  2, EParticleType::eFermion, false));
    inOutModelDef.defineParticleSpecies(ParticleDescription("BotL"         ,  1, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("BotR"         ,  3, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("tauL"         ,  4, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("tauR"         ,  5, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("CharmStrangeL",  6, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("CharmR"       ,  7, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("StrangeR"     ,  8, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("MuonL"        ,  9, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("MuonR"        , 10, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("UpDownL"      , 11, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("UpR"          , 12, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("DownR"        , 13, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("ElectronL"    , 14, EParticleType::eFermion, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("ElectronR"    , 15, EParticleType::eFermion, true));


    inOutModelDef.defineParticleSpecies(ParticleDescription("Gluon", 16, EParticleType::eBoson, false));
    inOutModelDef.defineParticleSpecies(ParticleDescription("W"    , 17, EParticleType::eBoson, false));
    inOutModelDef.defineParticleSpecies(ParticleDescription("B"    , 18, EParticleType::eBoson, true)); // hypercharge field

    inOutModelDef.defineParticleSpecies(ParticleDescription("Higgs"    , 19, EParticleType::eBoson, true));
    inOutModelDef.defineParticleSpecies(ParticleDescription("Goldstone", 20, EParticleType::eBoson, true));

}

// Computes propagator masses from action (Lagrangian) parameters. This example only includes thermal masses. Everything is in units of T.
wallgo::ModelParameters computeMasses(const wallgo::ModelParameters& actionParams)
{
    wallgo::ModelParameters outMsq;

    const double gs = actionParams.at("gs");
    const double gw = actionParams.at("gw");
    const double gY = actionParams.at("gY");

    // SU3 gluon
    outMsq.addOrModifyParameter("mg2", 2.0 * gs * gs);

    // FIXME dunno what these should be:
    
    // W boson
    outMsq.addOrModifyParameter("mw2", 1.0);
    
    // Bottom quark
    outMsq.addOrModifyParameter("mb2", 1.0);
    // Generic light quark
    outMsq.addOrModifyParameter("mq2", 1.0);

    // leptons
    outMsq.addOrModifyParameter("ml2", 1.0);

    // Higgs
    outMsq.addOrModifyParameter("mH2", 1.0);
    // "Goldstones"
    outMsq.addOrModifyParameter("mG2", 1.0);

    return outMsq;
}

void defineParametersSM(wallgo::ModelDefinition& inOutModelDef)
{
    wallgo::ModelParameters params;
    
    // COMMENT HERE: what are these params

    params.addOrModifyParameter("gs", 1.3);
    params.addOrModifyParameter("gw", 0.6);
    params.addOrModifyParameter("gY", 0.3);

    params.addOrModifyParameter("yt", 1.0);

    params.addOrModifyParameter("lam1H", 0.15);

    auto massSquares = computeMasses(params);

    inOutModelDef.defineParameters(params);
    inOutModelDef.defineParameters(massSquares);
}


int main() 
{
	std::cout << "### Running WallGo collision example : SM ###" << std::endl;

    wallgo::initializeRNG();

    wallgo::ModelDefinition modelDef;
    defineParametersSM(modelDef);
    defineParticlesSM(modelDef);

    wallgo::PhysicsModel model(modelDef);
    model.loadMatrixElements("MatrixElements/SM.json", /*print*/ false);
	
    // Use trivially small grid size to make the example run fast
	const int basisSizeN = 3;
    wallgo::CollisionTensor collisionTensor = model.createCollisionTensor(basisSizeN);


    wallgo::CollisionTensorVerbosity verbosity;
    verbosity.bPrintElapsedTime = true;
    verbosity.progressReportPercentage = 0;
    verbosity.bPrintEveryElement = true;

    // Test with very short Monte Carlo.
    wallgo::IntegrationOptions integrationOptions;
    integrationOptions.calls = 5000;
    integrationOptions.maxTries = 2;
    integrationOptions.maxIntegrationMomentum = 20;

    std::cout << "## Begin collision integrations ##" << std::endl;
    wallgo::CollisionTensorResult result = collisionTensor.computeIntegralsAll(integrationOptions, verbosity);

    return 0;
}
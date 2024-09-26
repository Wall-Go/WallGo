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
    const double lam1H = actionParams.at("lam1H");
    const double yt = actionParams.at("yt");

    // SU3 gluon
    outMsq.addOrModifyParameter("mg2",  gs * gs);

    
    // W boson
    outMsq.addOrModifyParameter("mw2", 11./12.*gw*gw);
    
    // U(1) boson
    outMsq.addOrModifyParameter("mb2", 11./12.*gY*gY);

    // Generic light quark - The mass is esimated as the asymptotic mass for a SU(3)_c fundamental fermion
    outMsq.addOrModifyParameter("mq2", gs*gs/3.);

    // leptons - The mass is esimated as the asymptotic mass for a SU(2)_L fundamental fermion
    outMsq.addOrModifyParameter("ml2", 3./16.*gw*gw);

    // Higgs - The mass is estimated as the one-loop thermal mass
    outMsq.addOrModifyParameter("mH2", 1./16.*(3*gw*gw+gY*gY+8*lam1H+4*yt*yt));
    // "Goldstones"- The mass is estimated as the one-loop thermal mass
    outMsq.addOrModifyParameter("mG2", 1./16.*(3*gw*gw+gY*gY+8*lam1H+4*yt*yt));

    return outMsq;
}

void defineParametersSM(wallgo::ModelDefinition& inOutModelDef)
{
    wallgo::ModelParameters params;
    

    /*
        Input values for pdg at mu=M_Z:

         Sw2=0.23129
         alpha_e=1/(127.944)
         alpha_s=0.1180
         G_F=1.16393
         m_t=172.57
         m_H=125.20

    */
    params.addOrModifyParameter("gs", 1.21772);
    params.addOrModifyParameter("gw", 0.651653);
    params.addOrModifyParameter("gY", 0.357449);
    params.addOrModifyParameter("yt", 1.00995);
    params.addOrModifyParameter("lam1H", 0.129008);

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